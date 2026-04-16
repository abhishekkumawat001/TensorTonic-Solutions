import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A = np.asarray(A)
    det = np.linalg.det(A) 
    if det == 0:
        return None
    if A.ndim == 2 or A.shape[0] == A.shape[i]:
        

        A_inv = np.linalg.inv(A)
    return A_inv 