import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    hinge_loss = np.maximum(0, (margin - y_score * y_true))
    if reduction=="mean":
        mean_hinge_loss = np.mean(hinge_loss)
    else:
        mean_hinge_loss = np.sum(hinge_loss)
        
    return float(mean_hinge_loss)