import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    
    if X.ndim != 2:
        return None

    N,D = X.shape
    
    if N <2 :
        return None
    X_centered = X - np.mean(X, axis = 0, keepdims = True)
    
    cov = 1/(N-1) * (X_centered.T @ X_centered)
    return cov