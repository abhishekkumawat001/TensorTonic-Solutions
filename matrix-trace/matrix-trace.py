import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    
    A = np.asarray(A)

    # Optional: check if square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None

    n = A.shape[0]
    trace_A = 0

    for i in range(n):
        trace_A += A[i, i]

    return trace_A