import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x, dtype=float)

    if rng is not None:
        r = rng.random(x.shape)
    else:
        r = np.random.random(x.shape)

    # Scaled dropout mask
    dropout_pattern = ((r >= p).astype(float)) / (1 - p)

    output = x * dropout_pattern

    return output, dropout_pattern