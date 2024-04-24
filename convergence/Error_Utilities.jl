function errors_mesh(F,Ftilde,dxdy)
    diff = F-Ftilde
    abs_err = norm.(diff)   # norm in case F[i,j] is of type vector
    rel_err = norm.(diff./F)

    l1   = sum(abs_err)*dxdy
    l2   = sqrt(sum(abs_err.^2)*dxdy)
    linf = maximum(abs_err)

    abs_errs = [l1;l2;linf];

    l1_rel   = sum(rel_err)*dxdy
    l2_rel   = sqrt(sum(rel_err.^2)*dxdy)
    linf_rel = maximum(rel_err)

    rel_errs = [l1_rel;l2_rel;linf_rel];
    
    return abs_errs, rel_errs, abs_err, rel_err
end
