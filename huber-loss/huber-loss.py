import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true = np.asarray(y_true) 
    y_pred = np.asarray(y_pred)
    e = y_true - y_pred
    e_abs = np.abs(e)
    loss = np.where(e_abs <= delta, 0.5 * e**2 , delta * (e_abs - 0.5 * delta) )
    
    
    loss2 = np.mean(loss)
    return float(loss2) 