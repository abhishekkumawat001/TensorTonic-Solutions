import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the SGNS loss.
    """
    # YOUR CODE HERE
    # Positive term: -log σ(v_c · u_o)
    pos_score = torch.dot(center_vec, pos_vec)
    pos_loss = F.softplus(-pos_score)

    # Negative terms: -Σ log σ(-(v_c · u_n))
    neg_scores = neg_vecs @ center_vec  # shape: (k,)
    neg_loss = F.softplus(neg_scores).sum()

    return pos_loss + neg_loss
