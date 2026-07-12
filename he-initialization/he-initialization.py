import numpy as np
def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    W = np.asarray(W)
    L = np.sqrt(6/fan_in)
    W_out = W * 2 * L  - L 
    return W_out