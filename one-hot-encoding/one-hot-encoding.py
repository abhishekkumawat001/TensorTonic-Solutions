import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    # Write code here
    y = np.asarray(y, dtype= int)

    if num_classes is None:
        num_classes = np.max(y) + 1

    if np.any(y <0) or np.any(y >= num_classes):
        raise VlaueError("Labels must be in the range[0, num_classes-1]")

    Y = np.zeros((len(y), num_classes), dtype = int)
    Y[np.arange(len(y)), y] = 1

    return Y