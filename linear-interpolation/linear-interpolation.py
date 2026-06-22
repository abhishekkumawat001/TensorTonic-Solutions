def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    result = values[:]
    n = len(result)

    i = 0
    while i < n:
        if result[i] is None:
            left = i - 1
            right = i

            while right < n and result[right] is None:
                right += 1

            left_val = result[left]
            right_val = result[right]
            span = right - left

            for j in range(left + 1, right):
                result[j] = left_val + (j - left) * (right_val - left_val) / span

            i = right
        else:
            i += 1

    return result