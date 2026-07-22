import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> tuple:
    """Compute input gate and candidate memory."""
    # YOUR CODE HERE
    con_dot = np.concatenate([h_prev,x_t], axis = -1)
    matmul_input = con_dot @ W_i.T + b_i
    
    i_t = sigmoid(matmul_input)

    matmul_candidate = con_dot @ W_c.T + b_c 

    candidate = np.tanh(matmul_candidate)

    return (i_t,candidate)
    