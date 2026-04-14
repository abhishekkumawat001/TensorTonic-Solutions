import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # Write code here
    x = np.asarray(x)
    
    pmf = []
    for xi in x:
        if xi == 1:
            prob = p 
        else:
            prob = 1-p
        pmf.append(prob)
    pmf = np.array(pmf)
    mean = p 
    var = p*(1-p)
    return pmf, mean , var