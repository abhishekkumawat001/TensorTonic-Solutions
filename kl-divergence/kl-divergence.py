import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p = np.asarray(p)
    q = np.asarray(q)

    kl_div = np.sum(p * np.log((p+eps)/ (q+eps)))

    return float(kl_div)
    