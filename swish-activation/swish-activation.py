import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    result = []
    x = np.asarray(x, dtype = float)
    for val in x:
        sig = 1 / (1 + np.exp(-val))
        swish = val * sig 
        result.append(swish)
    return np.array(result)