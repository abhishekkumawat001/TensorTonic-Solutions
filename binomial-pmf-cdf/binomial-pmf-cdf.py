import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    if k < 0 or k > n:
        return 0.0, 0.0

    pmf = comb(n,k) * (p**k) * ((1 - p)**(n - k))

    # cdf 
    cdf = 0.0
    for i in range(k + 1):
        cdf += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    return pmf , cdf
    