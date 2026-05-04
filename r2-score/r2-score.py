import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    nums = np.sum((y_true - y_pred)**2)
    deno = np.sum((y_true-np.mean(y_true))**2)
    
    if deno != 0:
        r_seq = 1 - nums / deno
    elif y_pred.all() == y_true.all():
        return 1 
    else:
        return 0

    return r_seq