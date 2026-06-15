import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    # Write code here
    X = np.asarray(X, dtype = float)
    x_mean = np.mean(X,axis,  keepdims = True)
    n_std = np.std(X,axis,  keepdims = True) + eps

    X = ( X - x_mean) / n_std

    return X