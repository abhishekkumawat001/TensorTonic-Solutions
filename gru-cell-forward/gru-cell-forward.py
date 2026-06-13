import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    input_size = params["Wz"].shape[0]
    hidden_size = params["Uz"].shape[0]
    x, x_was_1d = _as2d(x, input_size)
    h_prev, h_was_1d = _as2d(h_prev, hidden_size)
    
    z_t = _sigmoid(x @ params["Wz"] + h_prev @ params["Uz"] + params["bz"])
    r_t = _sigmoid(x @ params["Wr"] + h_prev @ params["Ur"] + params["br"])
    h_tilda = np.tanh(x @ params["Wh"] + (r_t * h_prev) @ params["Uh"] + params["bh"])
    h_t = (1-z_t) * h_prev + z_t * h_tilda

    if x_was_1d and h_was_1d:
        h_t = h_t.ravel() 
    return h_t