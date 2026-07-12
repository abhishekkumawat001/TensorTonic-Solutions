def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    W = np.asarray(W)
    
    L = np.sqrt(6/(fan_in+fan_out))

    w_dot = W * 2 * L - L 

    return w_dot