import numpy as np


def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    if N < 0:
        raise ValueError("N must be non-negative")
    if k <= 0 or k > N:
        raise ValueError("k must satisfy 1 <= k <= N")

    indices = np.arange(N)

    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)

    # Split into k folds with sizes differing by at most 1
    folds = np.array_split(indices, k)

    result = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        result.append((train_idx, val_idx))

    return result