import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    # Write code here
    x_t = cache[0]
    h_prev = cache[1]
    h_t = cache[2]
    W = cache[3]
    U = cache[4]
    
    dh = np.array(dh)
    h_t = np.array(h_t)
    x_t = np.array(x_t)
    h_prev = np.array(h_prev)
    W = np.array(W)
    U = np.array(U)
    #gradient through tanh
    dz = dh * (1-h_t**2)
    
    dx_t = W.T @ dz 
    dh_prev = U.T @ dz

    dW = np.outer(dz, x_t)
    dU = np.outer(dz, h_prev)
    db = dz
    return dx_t, dh_prev, dW, dU, db
