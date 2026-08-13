import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Write code here
    y = np.asarray(y)
    
    split_mask = np.asarray(split_mask, dtype=bool)

    # Parent entropy
    parent_entropy = _entropy(y)

    # Split into left and right children
    y_left = y[split_mask]
    y_right = y[~split_mask]

    n = y.size
    n_left = y_left.size
    n_right = y_right.size

    # If there are no samples, there is no information gain
    if n == 0:
        return 0.0

    # Weighted entropy after the split
    child_entropy = (
        (n_left / n) * _entropy(y_left)
        + (n_right / n) * _entropy(y_right)
    )

    # Information Gain
    return float(parent_entropy - child_entropy)