import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p = np.asarray(p)
    y = np.asarray(y)
    p = p.flatten()
    y = y.flatten()
    nums = (2 * np.sum(p*y)) + eps
    deno = np.sum(p) + np.sum(y) + eps

    dice_loss = 1 - nums / deno

    return dice_loss