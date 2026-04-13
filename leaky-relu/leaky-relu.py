import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x = np.asarray(x, dtype = float)
    result = []
    for x in x:
        if x >= 0:
            leaky_relu = x 
        else:
            leaky_relu = alpha * x 
        
        result.append(leaky_relu)
    
    return np.array(result)