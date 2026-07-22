import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> tuple:
    """Complete LSTM cell forward pass."""
    # YOUR CODE HERE
    con_dot = np.concatenate([h_prev,x_t], axis = -1) 
    matmul_t = con_dot @ W_f.T + b_f
    f_t = sigmoid(matmul_t)
    matmul_i = con_dot @ W_i.T + b_i
    i_t = sigmoid(matmul_i)
    matmul_c = con_dot @ W_c.T + b_c
    c_t = np.tanh(matmul_c)
    matmul_o = con_dot @ W_o.T + b_o
    o_t = sigmoid(matmul_o)
    candidate_t = f_t * C_prev + i_t * c_t
    h_t = o_t * np.tanh(candidate_t)

    return (h_t, candidate_t)