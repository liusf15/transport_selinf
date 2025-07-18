
def ci_bisection(get_pvalue, sd, right, left, sig_level=0.05, tol=1e-6):
    incre = sd / 5
    ## right end
    up = None
    pval_t = get_pvalue(right)
    if pval_t < sig_level:
        up = right
        t = right
        for i in range(200):
            t = t - incre
            pval_t = get_pvalue(t)
            if pval_t >= sig_level - tol:
                lo = t
                p_lo = pval_t
                up = t + incre
                break
        if i == 99:
            lo = right
            up = right + 10 * sd
    else:
        lo = right
        t = right
        for i in range(200):
            t = t + incre
            pval_t = get_pvalue(t)
            if pval_t <= sig_level + tol:
                up = t
                p_up = pval_t
                lo = t - incre
                break
    # bisection
    if up is None:
        rightend = float('inf')
    else:
        for i in range(200):
            if up - lo <= tol:
                break
            mid = (lo + up) / 2
            p_mid = get_pvalue(mid)
            if p_mid < sig_level:
                up = mid
            if p_mid >= sig_level:
                lo = mid
            if abs(p_mid - sig_level) <= tol:
                break
        if i == 99:
            raise AssertionError("did not converge")
        rightend = up
    
    ## left end
    lo = None
    pval_t = get_pvalue(left)
    if pval_t < sig_level:
        lo = left
        t = left
        for i in range(200):
            t = t + incre
            pval_t = get_pvalue(t)
            if pval_t >= sig_level - tol:
                up = t
                p_up = pval_t
                lo = t - incre
                break
        if i == 99:
            up = left
            lo = left - 10 * sd
    else:
        up = left
        t = left
        for i in range(200):
            t = t - incre
            pval_t = get_pvalue(t)
            if pval_t <= sig_level + tol:
                lo = t
                p_lo = pval_t
                up = t + incre
                break
    if lo is None:
        leftend = -float('inf')
    else:
        # bisection
        for i in range(200):
            if up - lo <= tol:
                break
            mid = (lo + up) / 2
            p_mid = get_pvalue(mid)
            if p_mid < sig_level:
                lo = mid
            if p_mid >= sig_level:
                up = mid
            if abs(p_mid - sig_level) <= tol:
                break
        if i == 99:
            raise AssertionError("did not converge")
        leftend = lo
    return (leftend, rightend)
