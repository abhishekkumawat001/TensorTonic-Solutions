import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.asarray(X , dtype = float)

    X_center = X - np.mean(X , axis = 0)

    C = (X_center.T @ X_center) /( X.shape[0] - 1 )

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]

    W = eigenvectors[:, idx[:k]]

    X_proj = X_center @ W

    return X_proj