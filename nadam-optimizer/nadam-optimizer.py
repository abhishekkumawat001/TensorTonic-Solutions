import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    # Write code here
    w = np.asarray(w, dtype = float)
    m = np.asarray(m, dtype = float)
    v = np.asarray(v, dtype = float)
    grad = np.asarray(grad, dtype = float)

    m_t = beta1 * m + (1-beta1) * grad 

    v_t = beta2 * v + (1-beta2) * grad ** 2

    w_t = w - lr * ( beta1 * m_t + (1-beta1)* grad) /( np.sqrt(v_t) + eps)

    return w_t , m_t , v_t 