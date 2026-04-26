import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)

    # Z1 = Z1 / np.linalg.norm(Z1, axis = 1 , keepdims = True )
    # Z2 = Z2 / np.linalg.norm(Z2, axis = 1 , keepdims = True)
    # (N, D) @ (D, N) -> (N, N)
    similarity_matrix = np.dot(Z1 , Z2.T) / temperature
    
    similarity_matrix_max = np.max(similarity_matrix, axis= 1 , keepdims = True)
    similarity_matrix_stable = similarity_matrix - similarity_matrix_max

    exp_s = np.exp(similarity_matrix_stable)
    
    positive_exp = np.diag(exp_s)

    denom = np.sum(exp_s, axis = 1)

    loss = - np.mean(np.log(positive_exp/denom))

    return loss

    
    