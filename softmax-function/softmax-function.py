import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x)
    x = x - np.max(x, axis = -1 , keepdims = True)
    nums = np.exp(x)
    deno = np.sum(nums, axis = -1, keepdims = True )
    soft = nums / deno 
    return soft 