import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    if not isinstance(matrix, (list, np.ndarray)):
        return None

    # Check rectangular shape BEFORE np.asarray
    try:
        row_lengths = [len(row) for row in matrix]
    except TypeError:
        return None  # not a 2D structure

    if len(set(row_lengths)) != 1:
        return None  # not rectangular

    matrix = np.asarray(matrix)
    N, D= matrix.shape 

    if matrix.ndim != 2 or N != D:
        return None
    if N == 0:
        return np.array([])
        
    eigenvalues = np.linalg.eigvals(matrix)

    idx = np.lexsort((eigenvalues.imag, eigenvalues.real))
    eigenvalues = eigenvalues[idx]

    return eigenvalues
    