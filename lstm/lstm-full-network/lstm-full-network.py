import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass. Returns (y, h_last, C_last).
        """
        # YOUR CODE HERE
        outputs = []
        B, T, D = X.shape
        h_0 = np.zeros((B,self.hidden_dim))
        c_0 = np.zeros((B,self.hidden_dim))
        for t in range(T):
            x_t = X[:,t,:]
            concatenate_h_x = np.concatenate([h_0, x_t], axis = -1)
            f_t = sigmoid(concatenate_h_x @ self.W_f.T + self.b_f)
            i_t = sigmoid(concatenate_h_x @ self.W_i.T + self.b_i)
            ctilda_t = np.tanh(concatenate_h_x @ self.W_c.T + self.b_c)
            o_t = sigmoid(concatenate_h_x @ self.W_o.T + self.b_o)
            c_t = f_t * c_0 + i_t * ctilda_t
            c_0 = c_t
            h_t = o_t * np.tanh(c_t)
            h_0 = h_t
        
            y_t = h_t @ self.W_y.T + self.b_y 
            outputs.append(y_t)
        outputs = np.stack(outputs, axis=1)
        return (outputs, h_t, c_t)
        