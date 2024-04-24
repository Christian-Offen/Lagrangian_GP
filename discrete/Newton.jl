function NewtonIter(F,z;TOL=1e-14,maxIter=100)
    
    i=0
    fval = F(z)
    DF(z) = ForwardDiff.jacobian(F,z)
    
    while (norm(fval)>TOL) && (i<maxIter)
        
        z = z - DF(z)\fval
        fval = F(z)
        i = i+1
    end
    
    return z, i
    
end 
