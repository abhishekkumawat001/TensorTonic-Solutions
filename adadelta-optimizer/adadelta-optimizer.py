import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    """
    # Write code here
    grad = np.asarray(grad, dtype = float)
    E_grad_sq = np.asarray(E_grad_sq, dtype = float)
    E_update_sq = np.asarray(E_update_sq, dtype = float)
    w = np.asarray(w,dtype = float)

    E_grad_sq_t = rho * E_grad_sq + (1-rho) * grad**2 
    num = np.sqrt(E_update_sq + eps)
    deno = np.sqrt(E_grad_sq_t + eps)

    parameter = - num / deno * grad 

    E_update_sq_t = rho * E_update_sq + (1-rho) * (parameter)**2

    w_t = w + parameter

    return w_t , E_grad_sq_t, E_update_sq_t