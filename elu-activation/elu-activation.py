import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    result = []
    x = np.asarray(x)

    for x in x:
        if x > 0:
            elu = x 
        else:
            elu = alpha * (np.exp(x) - 1 )
        result.append(elu)

    return result 