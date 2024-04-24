# 2jet of a function L(q,qdot)
function Jet2(L)

    L_q(q::Number,qdot::Number) = ForwardDiff.derivative(q->L(q,qdot),q)
    L_q(q,qdot) = ForwardDiff.gradient(q->L(q,qdot),q)
    
    L_qdot(q::Number,qdot::Number) = ForwardDiff.derivative(qdot->L(q,qdot),qdot)
    L_qdot(q,qdot) = ForwardDiff.gradient(qdot->L(q,qdot),qdot)
    
    L_qqdot(q::Number,qdot::Number) = ForwardDiff.derivative(qdot->L_q(q,qdot),qdot)
    L_qqdot(q,qdot) = ForwardDiff.jacobian(qdot->L_q(q,qdot),qdot)
    
    L_qdotq(q,qdot) = transpose(L_qqdot(q,qdot))
    
    L_qdotqdot(q::Number,qdot::Number) = ForwardDiff.derivative(qdot->L_qdot(q,qdot),qdot)
    L_qdotqdot(q,qdot) = ForwardDiff.jacobian(qdot->L_qdot(q,qdot),qdot)

	return L_q,L_qdot,L_qqdot, L_qdotq, L_qdotqdot
	
end


# Define EL, force, Energy
function EulerLagrange(L)
   
    L_q,L_qdot,L_qqdot, L_qdotq, L_qdotqdot = Jet2(L)
    
    # Euler-Lagrange
    #EL(q,qdot,qddot) = transpose(transpose(qdot)*L_qdotq(q,qdot)+transpose(qddot)*L_qdotqdot(q,qdot)-transpose(L_q(q,qdot)))
	
	EL(q,qdot,qddot) = L_qdotq(q,qdot)*qdot+L_qdotqdot(q,qdot)*qddot-L_q(q,qdot)
	
	function EL(qjet2)
		dimq=Int(length(qjet2)/3)
		return EL(qjet2[1:dimq],qjet2[dimq+1:2*dimq],qjet2[2*dimq+1:end])
	end


    # solve for acceleration
    acc(q,qdot) = L_qdotqdot(q,qdot) \ (-L_qdotq(q,qdot)*qdot + L_q(q,qdot))
    function acc(qjet)
    	dimq = Int(length(qjet)/2)
    	return acc(qjet[1:dimq],qjet[dimq+1:end])
    end	

    # Energy
    function H(q,qdot)
	    return transpose(qdot)*L_qdot(q,qdot) - L(q,qdot)
	end
    
    function H(qjet)
    	dimq = Int(length(qjet)/2)
    	return H(qjet[1:dimq],qjet[dimq+1:end])
    end	
    
    return EL, acc, H
end


# call to DifferentialEquations.solve
function DynamicsODEsolve(acc,u0,tspan; abstol=1e-8, reltol=1e-8)
    dim_q = Int(length(u0)/2)
    accl(y) = acc(y[1:dim_q],y[dim_q+1:end])
    f(u,p,t) = [u[dim_q+1:end]; accl(u)]
    prob = ODEProblem(f,u0,tspan,abstol=abstol, reltol=reltol)
    return sol = solve(prob)
end






