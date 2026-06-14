import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X, dtype=float)

    # Center each feature (column)
    X_centered = X - np.mean(X, axis=0)

    # Covariance matrix
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)

    # Standard deviations
    std_devs = np.std(X, axis=0, ddof=1)

    # Denominator matrix
    denom = np.outer(std_devs, std_devs)

    # Correlation matrix
    corr = cov / denom

    # Handle zero-variance features
    zero_var = (std_devs == 0)
    corr[zero_var, :] = np.nan
    corr[:, zero_var] = np.nan

    # Diagonal should be 1.0
    for i in range(len(std_devs)):
        if std_devs[i] != 0:
            corr[i, i] = 1.0

    return corr