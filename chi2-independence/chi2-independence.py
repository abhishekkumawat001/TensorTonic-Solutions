import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    c = np.array(C, dtype = float)
    row_sum = np.sum(c , axis = 1)
    col_sum = np.sum(c , axis = 0)
    total = np.sum(c)

    expected = np.outer(row_sum, col_sum) / total

    chi2_sum = np.sum((c-expected)**2 / expected)

    return float(chi2_sum) , expected