import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    batch_size, T, input_dim = X.shape #dim extract 
    h_t = h_0
    h_list = []
    
    for t in range(T):
        x_t = X[:, t, :]
        matrix_dot_prod = x_t @ W_xh.T + h_t @ W_hh.T + b_h
        h_t = np.tanh(matrix_dot_prod)
        h_list.append(h_t)
    
    hidden_states = np.stack(h_list, axis=1)  # (batch, T, hidden)
    h_final = h_t

    return hidden_states, h_final