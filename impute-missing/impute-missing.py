import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """

    X = np.asarray(X, dtype=float).copy()

    if strategy not in ('mean', 'median'):
        raise ValueError("strategy must be either 'mean' or 'median'")

    # Remember original shape
    was_1d = X.ndim == 1

    # Convert 1D -> 2D for column-wise processing
    if was_1d:
        X = X.reshape(-1, 1)

    # Process each column
    for j in range(X.shape[1]):
        col = X[:, j]

        valid = np.logical_not(np.isnan(col))

        # All values are NaN
        if not np.any(valid):
            col[:] = 0.0
            continue

        # Calculate statistic from valid values only
        if strategy == 'mean':
            value = np.mean(col[valid])
        else:
            value = np.median(col[valid])

        # Fill NaNs
        col[np.isnan(col)] = value

    # Return original dimensionality
    if was_1d:
        return X.ravel()

    return X