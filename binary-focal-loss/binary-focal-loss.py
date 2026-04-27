import math
def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    # Write code here
    predictions = list(predictions)
    targets = list(targets)
    total_loss = 0.0
    n = len(predictions)

    for p, y in zip(predictions, targets):
        # p_t depends on true class
        if y == 1:
            p_t = p
        else:
            p_t = 1 - p

        # Binary focal loss for one sample
        loss = -alpha * ((1 - p_t) ** gamma) * math.log(p_t)

        total_loss += loss

    # Mean loss over all samples
    return total_loss / n