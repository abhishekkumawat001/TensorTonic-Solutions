import torch
import torch.nn.functional as F

def cbow_forward(context_ids: torch.Tensor, target_id: int, W_in: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the CBOW cross-entropy loss for predicting target_id from the averaged context.
    """
    # YOUR CODE HERE\
    # 1. Average context embeddings
    h = W_in[context_ids].mean(dim=0)  # (embedding_dim,)

    # 2. Compute vocabulary scores (logits)
    z = W_out @ h  # (vocab_size,)

    # 3. Cross-entropy loss
    target = torch.tensor([target_id], device=z.device)
    loss = F.cross_entropy(z.unsqueeze(0), target)

    return loss
    
