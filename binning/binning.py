import numpy as np

def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    values = np.array(values)

    max_x = np.max(values)
    min_x = np.min(values)

    if max_x == min_x:
        return [0] * len(values)

    w = (max_x - min_x) / num_bins

    bins = np.minimum(((values - min_x) / w).astype(int), num_bins - 1)

    return bins.tolist()