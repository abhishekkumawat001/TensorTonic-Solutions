import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.asarray(a)
    b = np.asarray(b)
    deno = np.linalg.norm(a) * np.linalg.norm(b)
    if deno != 0:
        cos_similarity = np.dot(a , b) / deno
    else:
        return 0.0
    return float(cos_similarity)