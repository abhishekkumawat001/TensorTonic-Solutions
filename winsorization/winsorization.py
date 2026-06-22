import math

def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds.
    """
    if not values:
        return []

    arr = sorted(values)
    n = len(arr)

    def percentile(p):
        k = (n - 1) * p / 100.0
        lo = int(math.floor(k))
        hi = int(math.ceil(k))

        if lo == hi:
            return arr[lo]

        return arr[lo] + (k - lo) * (arr[hi] - arr[lo])

    lower_bound = percentile(lower_pct)
    upper_bound = percentile(upper_pct)

    return [
        lower_bound if x < lower_bound
        else upper_bound if x > upper_bound
        else x
        for x in values
    ]