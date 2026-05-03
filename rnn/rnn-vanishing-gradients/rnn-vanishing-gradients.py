import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
     # 1. Compute spectral norm (largest singular value)
    spectral_norm = np.linalg.norm(W_hh, ord=2)
    
    # 2. Initialize list with gradient at current step
    grad_norms = [1.0]

    for t in range(1 , T):
        next_norm = grad_norms[-1] * spectral_norm
        grad_norms.append(float(next_norm))
    
    return grad_norms