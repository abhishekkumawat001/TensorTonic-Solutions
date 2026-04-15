import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Write code here
    x = np.asarray(x)
    
    n = len(x)
    mean = np.mean(x)
    
    ssd = np.sqrt((np.sum((x - mean)**2)) / (n-1) )

    t_stats =  ( np.mean(x) - mu0 ) / ( ssd / np.sqrt(n))

    return t_stats 