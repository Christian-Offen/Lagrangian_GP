#   Learning of Lagrangian using a certified kernel-based method
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

using ForwardDiff
using Plots
using LinearAlgebra
#using DifferentialEquations
using NLsolve
using HaltonSequences
using BlockArrays
using SpecialFunctions
using Dates
using FileIO
using LaTeXStrings
using ProgressMeter

include("EL_utilities.jl");
include("L_learning_GP_utilities_ndim.jl");
include("VariationalMidpointRule.jl")

mNow() = string(Dates.format(now(), "yyyy-mm-dd_HH:MM:SS"));
println("Start: "*mNow())


dimQ=2
dimTQ = 2*dimQ;

## coupled harmonic oscillators
coupling_constant = 0.1
L_ref(q,qdot) = 1/2*sum(qdot.^2) - sum(q.^2) + coupling_constant*prod(q)

## make available on jet variables
L_ref(qjet) = L_ref(qjet[1:dimQ],qjet[dimQ+1:end])

EL_ref,acc_ref,H_ref = EulerLagrange(L_ref);

delta = 1
xx = range(-delta, delta, length=80)
yy = range(-delta, delta, length=80)

x=(a->[a;0]).(xx)
x2=(a->[0;a]).(xx)
y=(a->[a;0]).(yy)
y2=(a->[0;a]).(yy)

pcontourHref0=contour(xx, yy, H_ref.(x',y),xlabel=L"x^0",ylabel=L"\dot{x}^0")
plot!(size=(300,200)); savefig(pcontourHref0,"plots/ContourHref0_"*mNow()*".pdf")

u0_0 = [0.2, 0.1, 0., 0.]
tspan_0 = (0.0,100.0)
h_ref0 = 0.01
tt0 = tspan_0[1]:h_ref0:tspan_0[end]
steps0 = Int(tspan_0[end]/h_ref0)

sol_ref = ComputeTrajectory(LdMP(L_ref,h_ref0),u0_0[1:dimQ],u0_0[dimQ+1:end],steps0;ftol=1e-8);

pSolRef0=plot(tt0,sol_ref', label=[L"x^0" L"x^1"],xlabel=L"t")
plot!(size=(300,200)); savefig(pSolRef0,"plots/SolRef0_"*mNow()*".pdf")

#   Data generation
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# samples and domain
NSamples = 300
domain_bnd = [1., 1.,1.,1.]
basept = zeros(4)

bnd_value = [1., 1., 1.] # normalisation of conj. momentum of L and value of L at base point

samples_per_unit = NSamples/prod(domain_bnd*2)
println("samples per unit: "*string(samples_per_unit))

data_TQ=HaltonPoint(2*dimQ,length=NSamples)
data_TQ=(a->domain_bnd.*(-ones(2*dimQ) .+ 2.0*a)).(data_TQ)

accl_ref(y::Vector) = acc_ref(y[1:dimQ],y[dimQ+1:end])
data_TQ2 = ((a,b) -> cat(dims=1,a,b)).(data_TQ,accl_ref.(data_TQ));

Data=hcat(data_TQ2...);

#predicted size of linear problem in training / predictions
size_theta = dimQ*(NSamples+1)+1
println("predicted size of linear problem "*string(size_theta)*"x"*string(size_theta))

#   kernel definition
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# kernel definition
lengthscale = 1*ones(dimTQ); #[0.9; 0.9; 0.9; 0.9]

# squared exponential
kernel(a,b) = exp(-0.5*sum(((a-b)./lengthscale).^2))
kernel(q1,q1dot,q2,q2dot) = kernel([q1;q1dot],[q2;q2dot])

vol_n_ball = pi^(dimTQ/2)/gamma(dimTQ/2+1)*prod(lengthscale)
samples_per_lengthscale_ball = vol_n_ball*samples_per_unit
println("# samples within length scale: "*string(samples_per_lengthscale_ball))


#   Computation of conditional mean and covariance operator
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

println("Define conditional mean L_ml and variances "*mNow())

L_ml, var_ml_operator, var_ml_pts, var_ml_p, var_ml_H, Theta = Lagrangian_Gamblet(kernel,data_TQ2,basept; normalisation_balancing=bnd_value);
println(mNow())


#   Evaluation
#   ============

# dynamics prediction from initial data of sol_ref
tspan_0 = (0.,100.)

println("Compute dynamics "*mNow())
sol1_ml = DELSolve(LdMP(L_ml,h_ref0),sol_ref[:,1],sol_ref[:,2],steps0;ftol=1e-8);

println("Plotting/Saving of results "*mNow())

psol1_ml=plot(tt0,sol1_ml', label=[L"x^0" L"x^1"])
plot!(size=(300,200))
savefig(psol1_ml,"plots/psol1_ml_"*mNow()*".pdf")

pcompareDyn=plot(tt0,sol_ref', label="ref",xlabel=L"t",color=:darkgray,linestyle=:dash)
plot!(tt0,sol1_ml', label=[L"x^0" L"x^1"])
plot!(size=(300,200))
savefig(pcompareDyn,"plots/pcompareDyn_MP_"*mNow()*".pdf")

pcompareDynx0=plot(tt0,sol_ref[1,:], label="ref",xlabel=L"t",ylabel=L"x^0",color=:darkgray)
plot!(tt0,sol1_ml[1,:], label="ml")
plot!(size=(300,200))
savefig(pcompareDynx0,"plots/pcompareDynx0_MP_"*mNow()*".pdf")

# save ode solution / energy / variance
save("OdeSolution_MP_"*mNow()*".jld2","sol_ref",sol_ref,"sol1_ml",sol1_ml)
