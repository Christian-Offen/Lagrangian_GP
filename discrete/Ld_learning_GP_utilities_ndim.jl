function ExplHandles(L,dim_L)
    return [x->L(x)[j] for j=1:dim_L]
end


# Euler-Lagrange for dimQ
function del(Ld,qjet::Vector)

	dimQ  = Int(length(qjet)/3)
	q0 = qjet[1:dimQ]	
	q1 = qjet[dimQ+1:2*dimQ]
	q2 = qjet[2*dimQ+1:end]
	
	LD(q0,q1) = Ld([q0;q1])
	
	DEL = -ConjugateMomenta_p0(LD,q1,q2) + ConjugateMomenta_p1(LD,q0,q1)
    
    return DEL
end

function del(L,qjet,dim_L)
    return hcat(Base.Fix2(del,qjet).(ExplHandles(L,dim_L))...)
end
    
function conjp(Ld,qjet2)
	LD(q0,q1) = Ld([q0;q1])
    return ConjugateMomenta_p0(LD,qjet2)
end
   
function conjp(L,qjet2,dim_L)
    return hcat(Base.Fix2(conjp,qjet2).(ExplHandles(L,dim_L))...)
end


# linear observation functionals Phi 
# evaluation of EL operator and conjp operator at data points and base
# element in dual space
function ObservationFunct(Ld,data_QQQ,basept)
    
    delL_vals = Base.Fix1(del,Ld).(data_QQQ)
    conjpL_vals = conjp(Ld,basept);
    base_eval = Ld(basept)
    
    return vcat([delL_vals; conjpL_vals; base_eval]...)
end


function flattenBlockMat(M)
    return Matrix(mortar(M))
end

function theta_k(kernel,data_TQ2,basept) 

    elel(a,b)= del(a->del( b_ -> kernel(a,b_),b),a,dimQ)
    elp(a,b) = del(a_->conjp(b_ -> kernel(a_,b_),b),a,dimQ)
    pel(a,b) = conjp(a_->del( b_ -> kernel(a_,b_),b),a,dimQ)
    pp(a,b) = conjp(a_->conjp( b_ -> kernel(a_,b_),b),a,dimQ)
    
    elev(a,b) = del(a_->kernel(a_,b),a)
    pev(a,b) = conjp(a_->kernel(a_,b),a)
    
    # evaluations
    #elelM = elel.(data_TQ2,data_TQ2')
    elelM = [elel(a,b) for a in data_QQQ, b in data_QQQ]
    elpM = Base.Fix2(elp,basept).(data_TQ2)
    pelM = Base.Fix1(pel,basept).(data_TQ2)
    ppM = pp(basept,basept)
    
    elevM = Base.Fix2(elev,basept).(data_TQ2)
    pevM = pev(basept,basept)
    evevM = kernel(basept,basept)
    
    theta = [flattenBlockMat(elelM) vcat(elpM...) vcat(elevM...); hcat(pelM...) ppM pevM; vcat(elevM...)' pevM' evevM]
end

# theta_k(kernel,data_TQ2,basept) = hcat([ObservationFunct(b->ObservationFunct(a->kernel(a,b),data_TQ2,basept)[j],data_TQ2,basept) for j=1:size_theta]...) # alternative, slower implementation of theta_k(kernel,data_TQ2,basept)

function kappaPhi(kernel,data_TQ2,basept)

    el_prep(a,b)=  del( b_ -> kernel(a,b_),b)
    elM(a) = Base.Fix1(el_prep,a).(data_TQ2)
    
    p_prep(a,b)=  conjp( b_ -> kernel(a,b_),b)
    pM(a) = p_prep(a,basept)

    return a->[vcat(elM(a)...); pM(a); kernel(basept,a)]
    
end


# Relation to gamblets: Prop2.1 Chen, Hosseini, Owhadi, Stuart: Solving and learning nonlinear PDEs with Gaussian Processes
function Lagrangian_Gamblet(kernel,data_TQ2,basept;normalisation_balancing=ones(dimQ+1))
    
    Theta = theta_k(kernel,data_TQ2,basept)
    Theta_fact=factorize(Theta)
    
    KappaPhi = kappaPhi(kernel,data_TQ2,basept) # Element in RKHS. This is a function handle.
    
    
    # Lagrangian as conditional mean
    function L_ml(qjet)
        gamblets_values = Theta_fact\KappaPhi(qjet)
        return normalisation_balancing'*gamblets_values[end-dimQ:end]
    end
    
    function L_ml(q,qdot)
        return L_ml([q; qdot])
    end
    
    # CONDITIONAL VARIANCE
    # covariance operators for various observables
    
    # variance for observation variable operator; operator(#1)(qjet) \in U^*     (U^* = dual of RKHS)
    function var_ml_operator(operator)
        
        function var_ml(qjet)
        
            operator_handle(b) = operator(a->kernel(a,b))(qjet)
            kappa_Phi_phi = ObservationFunct(operator_handle,data_TQ2,basept)
            linSys = Theta_fact\kappa_Phi_phi

            kappa_phi_phi = operator(operator_handle)(qjet)

            return kappa_phi_phi - kappa_Phi_phi'*linSys
        end
        
        function var_ml(q,qdot)
            return var_ml([q;qdot])
        end
        
        return var_ml
        
    end
    
    
    var_ml_pts = var_ml_operator(L-> (x ->L(x)))        # point evaluation
    
    # conjugate momenta
    var_ml_pA=var_ml_operator.([L-> (x -> conjp(L,x)[j]) for j=1:dimQ])
    var_ml_P(pt)= [var_ml_pA[j](pt) for j=1:dimQ]
    
    #var_ml_H   = var_ml_operator(HamOperator)           # energy evaluation
    
        
    return L_ml, var_ml_operator, var_ml_pts, var_ml_P, Theta

end
