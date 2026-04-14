import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x = np.asarray(x)

    mean = np.mean(x)
    median = np.median(x)
    counts = Counter(x)
    max_freq = max(counts.values())
    mode = min([k for k, v in counts.items() if v == max_freq])

    
    return float(mean), float(median), float(mode)