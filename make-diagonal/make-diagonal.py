import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    v = np.asarray(v)
    n = len(v)
    A = np.zeros((n, n))

    for i in range(n):
        A[i, i] = v[i]
    return A 