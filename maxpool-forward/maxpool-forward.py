import numpy as np 

def maxpool_forward(X , pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    X = np.asarray(X)
    H, W = X.shape 

    #output dims 
    h_out = (H - pool_size)//stride + 1
    w_out = (W - pool_size)//stride + 1 

    output = np.zeros((h_out, w_out))

    # slide the pooling window 
    for i in range(h_out):
        for j in range(w_out):
            max_val = float("-inf")

            # traverse the pooling window 
            for a in range(pool_size):
                for b in range(pool_size):
                    val = X[i * stride + a ][j * stride + b ]
                    if val > max_val:
                        max_val = val 
            output[i][j] = max_val

    return output.tolist()