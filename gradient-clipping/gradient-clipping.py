import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.asarray(g)
    norm = np.linalg.norm(g, keepdims = False)

    if norm == 0 or norm <= max_norm:
        return g.copy() 
    elif max_norm <= 0:
        return g.copy() 
    else:
        return g * (max_norm/norm)
        