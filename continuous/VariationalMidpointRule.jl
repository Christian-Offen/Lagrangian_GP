# include("Newton.jl")

function LdMP(L,h)
	Ld(q0,q1) = h*L((q0+q1)./2,(q1-q0)./h)
	return Ld
end


function DELSolve(Ld,q0,q1;ftol=1e-8,maxIterNewton=100)
    q2guess = 2*q0-q1
    DELObjective(q2) = ForwardDiff.gradient(q1->Ld(q0,q1)+Ld(q1,q2),q1)
    return nlsolve(DELObjective,q2guess,autodiff = :forward,ftol=ftol)
    #return NewtonIter(DELObjective,q2guess;TOL=ftol,maxIter=maxIterNewton)
end

function DELSolve(Ld,q0,q1,steps;ftol=1e-8,maxIterNewton=100)

    trj = zeros((size(q0)[1],steps+1))
    trj[:,1]=q0
    trj[:,2]=q1
    
    @showprogress for j = 1:steps-1        
        trj[:,j+2] = DELSolve(Ld,trj[:,j],trj[:,j+1],ftol=ftol).zero
#        trj[:,j+2] = DELSolve(Ld,trj[:,j],trj[:,j+1],ftol=ftol,maxIterNewton=maxIterNewton)[1]
    end
    
    return trj
end

function ConjugateMomenta_p0(Ld,q0,q1)
    return -ForwardDiff.gradient(q0->Ld(q0,q1),q0)
end

function ConjugateMomenta_p1(Ld,q0,q1)
    return ForwardDiff.gradient(q1->Ld(q0,q1),q1)
end

function ConjugateMomenta_p0(Ld,q0q1)
	d = Int(length(q0q1)/2)
	return ConjugateMomenta_p0(Ld,q0q1[1:d],q0q1[d+1:end])
end

function ConjugateMomenta_p1(Ld,q0q1)
	d = Int(length(q0q1)/2)
	return ConjugateMomenta_p1(Ld,q0q1[1:d],q0q1[d+1:end])
end

function ConjugateMomenta_p0(Ld,trj::Matrix)
	q0q1=[trj[:,1:end-1]; trj[:,2:end]]
	ps = mapslices(Base.Fix1(ConjugateMomenta_p0,Ld),q0q1,dims=1)
	p_N = ConjugateMomenta_p1(Ld,q0q1[:,end])
	return [ps p_N]
end

function ConjugateMomenta_p1(Ld,trj::Matrix)
	q0q1=[trj[:,1:end-1]; trj[:,2:end]]
	ps = mapslices(Base.Fix1(ConjugateMomenta_p1,Ld),q0q1,dims=1)
	p0 = ConjugateMomenta_p0(Ld,q0q1[:,1])
	return [p0 ps]
end



function Step1(Ld,q0,p0;ftol=1e-8)
    #q1guess = zeros(size(q0)[1])
    q1guess = q0
    objective = q1 -> p0-ConjugateMomenta_p0(Ld,q0,q1)
    return nlsolve(objective,q1guess,autodiff =:forward,ftol=ftol)
    #return NewtonIter(objective,q1guess,TOL=ftol)
end

function ComputeTrajectory(Ld,q0,p0,steps;ftol=1e-8,maxIterNewton=100)
    q1 = Step1(Ld,q0,p0).zero
#    q1 = Step1(Ld,q0,p0)[1]
    return DELSolve(Ld,q0,q1,steps,ftol=ftol,maxIterNewton=maxIterNewton)
end



