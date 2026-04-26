import numpy as np
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1 = list(x1)
    x2 = list(x2)
    x1 = x1 / np.linalg.norm(x1 ,  keepdims = True)
    x2 = x2 / np.linalg.norm(x2 ,  keepdims = True)
    cos_simi = np.dot(x1, x2)
    if label == 1:
        loss = 1 - cos_simi
    else: 
        loss =  max(0 , (cos_simi-margin))

    return float(loss) 