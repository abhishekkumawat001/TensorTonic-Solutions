import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    if lam <= 0 or k < 0 :
        raise ValueError("lam must be > 0  and k must be >= 0")

    factorial = np.prod(np.arange(1, k+1)) if k > 0 else 1
    
    pmf = (np.exp(-lam)) * (lam**k) / factorial

    cdf = 0 
    factorial = 1

    for i in range(0, k+1):
        if i > 0:
            factorial *= i
        cdf += np.exp(-lam) * (lam**i) / factorial

    return pmf , cdf 

    