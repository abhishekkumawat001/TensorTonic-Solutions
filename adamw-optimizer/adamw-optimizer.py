import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    # Write code here
    w = np.asarray(w, dtype = float)
    m = np.asarray(m, dtype = float)
    v = np.asarray(v, dtype = float)
    grad = np.asarray(grad, dtype = float)
    
    m_t = beta1 * m + (1-beta1) * grad

    v_t = beta2 * v + (1- beta2) * grad**2 

    w_t = w - lr * ( weight_decay * w) - lr * m_t * (1 / (v_t**0.5 + eps))

    return (w_t, m_t, v_t)