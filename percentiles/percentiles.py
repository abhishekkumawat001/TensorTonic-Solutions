import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    
    x = np.sort(x)
    n = len(x)

    result = []

    for p in q:
        L = (p / 100) * (n - 1)

        # Case 1: exact index
        # if float(L).is_integer():
        #     idx = int(L) - 1
        #     idx = min(max(idx, 0), n - 1)
        #     result.append(x[idx])
        #     continue

        # Case 2: interpolation
        lower = int(np.floor(L))
        upper = int(np.ceil(L))

        if upper == lower:
            result.append(x[lower])
            continue

        fraction = L - lower
        value = x[lower] + fraction * (x[upper] - x[lower])

        result.append(value)

    return np.asarray(result)