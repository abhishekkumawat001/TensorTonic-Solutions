import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A, dtype = float)
    rows, cols = A.shape
    # print(A.shape)
    # print(rows,cols)
    results = np.zeros((cols,rows), dtype = float)
    # print(results)
    for i in range(rows):
        for j in range(cols):
            results[j,i] = A[i,j]

    return results
