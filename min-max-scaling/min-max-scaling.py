import numpy as np
def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    data = np.asarray(data , dtype = float)
    
    n_rows = len(data)
    n_cols = len(data[0])

    # Create result matrix
    result = [[0.0] * n_cols for _ in range(n_rows)]

    for j in range(n_cols):
        column = [data[i][j] for i in range(n_rows)]

        min_val = min(column)
        max_val = max(column)
        range_val = max_val - min_val

        if range_val == 0:
            # Constant column -> all zeros
            for i in range(n_rows):
                result[i][j] = 0.0
        else:
            for i in range(n_rows):
                result[i][j] = (data[i][j] - min_val) / range_val

    return result
