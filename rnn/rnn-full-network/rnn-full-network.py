import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        # YOUR CODE HERE
        batch, seq , inputdim = X.shape
        if h_0 is None:
            h_t = np.zeros((batch, self.hidden_dim))
        else:
            h_t = h_0 
        h_list = []
        y_list = []
        
        for t in range(seq):
            x_t = X[:, t , :]
            #hidden state (update)
            h_t = np.tanh(x_t @ self.W_xh.T + h_t @ self.W_hh.T + self.b_h)
            # output (important matrix)
            y_t = h_t @ self.W_hy.T + self.b_y
            
            h_list.append(h_t)
            y_list.append(y_t)
            
        y_seq = np.stack(y_list , axis = 1) 
        hidden_states= np.stack(h_list, axis = 1)

        h_final = h_t
        return y_seq , h_final     