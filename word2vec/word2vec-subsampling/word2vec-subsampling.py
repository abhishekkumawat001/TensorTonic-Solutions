import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    # YOUR CODE HERE
    counts = counts.float()
    N = counts.sum()
    freqs = counts / N

    keep_probs = torch.minimum(
        torch.ones_like(freqs),
        torch.sqrt(torch.tensor(t, dtype=freqs.dtype, device=freqs.device) / freqs)
    )

    return keep_probs    
