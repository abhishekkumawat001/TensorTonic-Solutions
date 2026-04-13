import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    result = []
    x = np.asarray(x)
    for x in x:
        tanh = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        result.append(tanh)
    return result