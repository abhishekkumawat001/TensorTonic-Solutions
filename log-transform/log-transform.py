import numpy as np
def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # Write code here
    values = np.asarray(values, dtype = float)

    return np.log(values + 1 )