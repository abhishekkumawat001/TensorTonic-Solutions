import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.asarray(y)

    values, counts = np.unique(y, return_counts = True)
    y = counts / counts.sum()
    y = y[y>0]
    ln = np.log2(y)
    pln = y*ln 
    hs = - np.sum(pln)
    return hs 