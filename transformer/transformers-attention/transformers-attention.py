import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_k = Q.size(-1)

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Scale the scores
    scores = scores / math.sqrt(d_k)

    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Compute weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output