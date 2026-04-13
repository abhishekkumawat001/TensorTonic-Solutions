import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.array(x, dtype = float)
    result = []
    erf_vec = np.vectorize(math.erf)
    for val in x:
        gelu = 0.5* val*( 1 + (erf_vec(val/2**0.5)))
        result.append(gelu)
    return np.array(result)    
    
