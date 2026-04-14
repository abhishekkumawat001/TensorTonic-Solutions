import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    # Write code here
    k = np.asarray(k)
    pmf = []
    
    for i in k:
        pm = (1-p)**(i-1) * (p)
        pmf.append(pm)
    
    mean = 1/p

    return np.array(pmf) , mean