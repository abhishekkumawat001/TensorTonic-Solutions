def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    if len(values) == 0:
        return []

    if len(values) == 1:
        return [0.0]
        
    vals = sorted(values)
    n = len(vals)

    def median(arr):
        m = len(arr)
        if m % 2 == 1:
            return arr[m // 2]
        else:
            return (arr[m // 2 - 1] + arr[m // 2]) / 2

    # Median
    med = median(vals)

    # Lower and upper halves
    if n % 2 == 1:
        lower = vals[:n // 2]          # exclude median
        upper = vals[n // 2 + 1:]
    else:
        lower = vals[:n // 2]
        upper = vals[n // 2:]

    q1 = median(lower)
    q3 = median(upper)

    iqr = q3 - q1

    if iqr == 0:
        return [0.0] * n

    return [((x - med) / iqr) for x in values]