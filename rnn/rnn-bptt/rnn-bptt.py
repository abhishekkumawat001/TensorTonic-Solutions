import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    # YOUR CODE HERE
    # 1. Backprop through tanh
    dtanh = (1 - h_t**2) * dh_next   # (batch, hidden)

    # 2. Gradient w.r.t previous hidden state
    dh_prev = dtanh @ W_hh           # (batch, hidden)

    # 3. Gradient w.r.t hidden-to-hidden weights
    dW_hh = dtanh.T @ h_prev         # (hidden, hidden)

    return dh_prev, dW_hh