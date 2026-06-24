import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor, center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> tuple:
    """
    Returns tuple (W_in_updated, W_out_updated), each the same shape as the inputs, after one SGNS SGD step.
    """
    # YOUR CODE HERE
    W_in = W_in.clone()
    W_out = W_out.clone()

    # Pre-update vectors
    v_c = W_in[center_id].clone()
    u_o = W_out[pos_id].clone()
    u_negs = W_out[neg_ids].clone()

    # Scores
    s_o = torch.dot(v_c, u_o)
    s_negs = u_negs @ v_c

    # Sigmoids
    sig_o = torch.sigmoid(s_o)
    sig_negs = torch.sigmoid(s_negs)

    # Gradients for output embeddings
    grad_u_o = (sig_o - 1.0) * v_c
    grad_u_negs = sig_negs.unsqueeze(1) * v_c.unsqueeze(0)

    # Gradient for center embedding
    grad_v_c = (sig_o - 1.0) * u_o + (sig_negs.unsqueeze(1) * u_negs).sum(dim=0)

    # SGD updates
    W_in[center_id] -= lr * grad_v_c
    W_out[pos_id] -= lr * grad_u_o

    for i, neg_id in enumerate(neg_ids):
        W_out[neg_id] -= lr * grad_u_negs[i]

    return W_in, W_out
