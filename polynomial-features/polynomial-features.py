def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    # Write code here
    result = []

    for x in values:
        row = [x**p for p in range(0 , degree+1)]
        result.append(row)

    return result 