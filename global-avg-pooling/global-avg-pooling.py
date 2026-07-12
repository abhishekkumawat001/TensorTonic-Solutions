import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x = np.asarray(x)

    if x.ndim == 3:          # (C, H, W)
        return x.mean(axis=(1, 2)).tolist()
    elif x.ndim == 4:        # (N, C, H, W)
        return x.mean(axis=(2, 3)).tolist()
    else:
        raise ValueError("Input must have shape (C, H, W) or (N, C, H, W)")