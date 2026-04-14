import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.asarray(x)
    n = len(x)
    var = 1/(n-1) * (np.sum((x - np.mean(x))**2))

    std_var = np.sqrt(var)

    return var , std_var