import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.asarray(x, dtype = float)
    z = 1 + np.exp(-x)
    y = 1/z
    return y