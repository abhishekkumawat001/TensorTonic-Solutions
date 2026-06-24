import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    # YOUR CODE HERE
    
    n = len(token_ids)
    pairs = []

    for i in range(n):
        left = max(0, i - window)
        right = min(n - 1, i + window)

        for j in range(left, right + 1):
            if j != i:
                pairs.append([int(token_ids[i]), int(token_ids[j])])

    if not pairs:
        return torch.empty((0, 2), dtype=torch.int64)

    return torch.tensor(pairs, dtype=torch.int64)