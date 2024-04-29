#   Learning of Lagrangian as gamblets using a certified kernel-based method
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

using DoubleFloats

using ForwardDiff
using Plots
using LinearAlgebra
#using DifferentialEquations
using HaltonSequences
using BlockArrays
using SpecialFunctions
using Dates
using FileIO

#using DoubleFloats, ForwardDiff, Plots, LinearAlgebra, HaltonSequences, BlockArrays, SpecialFunctions, Dates, FileIO

include("EL_utilities.jl");
include("L_learning_GP_utilities_ndim.jl");
include("Error_Utilities.jl");

# Data sizes for convergence test
DataSizes = (2).^(1:15);

# use below for testing memory requirements in advance
# NSamples=2^15; size_theta = 1*(NSamples+1)+1; println("Memory requirement: "*string(Base.format_bytes(size_theta*size_theta*16)))


println("Data sizes: "*string(DataSizes))

mNow() = string(Dates.format(now(), "yyyy-mm-dd_HH:MM:SS"));
println("Start: "*mNow())

#   Reference system / true system
#   ================================

dimQ=1
dimTQ = 2*dimQ;

## harmonic oscillators
L_ref(q,qdot) = 1/2*sum(qdot.^2) - sum(q.^2) # + coupling_constant*prod(q)

## make available on jet variables
L_ref(qjet) = L_ref(qjet[1:dimQ],qjet[dimQ+1:end])

EL_ref,acc_ref,H_ref = EulerLagrange(L_ref);

delta = DoubleFloat(1)
xx = range(-delta, delta, length=10)
yy = range(-delta, delta, length=11)

x=xx
y=yy

# area of mesh element
dxdy = (xx[2]-xx[1])*(yy[2]-yy[1]);

#   Data generation
#   =================

# samples and domain
NSamples = maximum(DataSizes)
domain_bnd = DoubleFloat.([1., 1.])
basept = zeros(DoubleFloat,2)

bnd_value = DoubleFloat.([1., 1.]) # normalisation of conj. momentum of L and value of L at base point

samples_per_unit = NSamples/prod(domain_bnd*2)
println("samples per unit: "*string(samples_per_unit))

data_TQ=HaltonPoint(2*dimQ,length=NSamples)
data_TQ=(a->domain_bnd.*(-ones(2*dimQ) .+ 2.0*a)).(data_TQ)

accl_ref(y::Vector) = acc_ref(y[1:dimQ],y[dimQ+1:end])
data_TQ2 = ((a,b) -> cat(dims=1,a,b)).(data_TQ,accl_ref.(data_TQ));

Data=hcat(data_TQ2...);

#predicted size of linear problem in training / predictions
size_theta = dimQ*(NSamples+1)+1
bits_entry = Base.summarysize(DoubleFloat(0))
println("predicted max size of linear problem "*string(size_theta)*"x"*string(size_theta))
println("Memory requirement: "*string(Base.format_bytes(size_theta*size_theta*bits_entry)))

pDataScatter=scatter(Data[1,:],Data[2,:],label="collocation pts",markersize=2)
scatter!(basept[1:1],basept[2:2], markershape=:xcross, markersize=8, markerstrokewidth=5, label="bnd. cond")
contour!(xx, yy, H_ref.(x',y),xlabel="q1",ylabel="q1'")

savefig(pDataScatter,"plots/DataScatter_"*mNow()*".pdf")

#   kernel definition
#   ===================

# kernel definition
lengthscale = 1*ones(DoubleFloat,dimTQ); #[0.9; 0.9; 0.9; 0.9]

# squared exponential
kernel(a,b) = exp(-sum(((a-b)./lengthscale).^2)/2)
kernel(q1,q1dot,q2,q2dot) = kernel([q1;q1dot],[q2;q2dot])

vol_n_ball = pi^(dimTQ/2)/gamma(dimTQ/2+1)*prod(lengthscale)
samples_per_lengthscale_ball = vol_n_ball*samples_per_unit
println("# samples within length scale: "*string(samples_per_lengthscale_ball))

#   Computation of gamblet
#   ========================

Errors = zeros(DoubleFloat,length(DataSizes),6);
maxVarH = zeros(DoubleFloat,length(DataSizes));
acc_ref_val = acc_ref.(x',y);

for (index,maxData) in enumerate(DataSizes)
    L_ml, _ , _ , _ , var_ml_H , _ = Lagrangian_Gamblet(kernel,data_TQ2[1:maxData],basept; normalisation_balancing=bnd_value);

    println("Data size "*string(maxData))
    
    # further operator
    acc_ml = qjet-> acc(L_ml,qjet);

    # for convenient evaluations on meshes
    acc_ml_vec(a,b) = acc_ml([a;b])[1]
    var_ml_Ham_vec(a,b) = var_ml_H([a;b])

    # test how well accelleration matches reference
    acc_ml_val = acc_ml_vec.(x',y)

    # errors acceleration
    abs_errs, rel_errs, _,_ = errors_mesh(acc_ref_val,acc_ml_val,dxdy);
    Errors[index,:] = vcat(abs_errs, rel_errs)

    # check varHam
    maxVarH[index] = maximum(abs.(var_ml_Ham_vec.(x',y)))
    
    # save intermediate result to file
    save("computations_results_run/Error_Values_Iter"*string(index)*"_"*mNow()*".jld2","Errors",Errors,"maxVarH",maxVarH)

end

    

pAbsErr=plot(DataSizes,Errors[:,1:3], xaxis=:log, yaxis=:log, xlabel="data size",label=["l1" "l2" "linf"],title="absolute error acceleration")
savefig(pAbsErr,"plots/AbsErr_"*mNow()*".pdf")

pRelErr=plot(DataSizes,Errors[:,4:6], xaxis=:log, yaxis=:log, xlabel="data size",label=["l1" "l2" "linf"],title="relative error acceleration")
savefig(pRelErr,"plots/RelErr_"*mNow()*".pdf")

pMaxVarH=plot(DataSizes,maxVarH, xaxis=:log, yaxis=:log, xlabel="data size",label="max(varH)",title="variance H")
savefig(pMaxVarH,"plots/MaxVarH_"*mNow()*".pdf")

println("End of script "*mNow())

#using NBInclude
#nbexport("L_Learning_CertifiedGP_oscillator_1d_convergence.jl", "L_Learning_CertifiedGP_oscillator_1d_convergence.ipynb")
