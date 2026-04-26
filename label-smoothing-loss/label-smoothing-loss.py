import math
def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    # Write code here
    predictions = list(predictions)
    k = len(predictions)

    loss = 0.0
    
    for i in range(k):
        if i == target:
            q = (1-epsilon) + epsilon / k 
        else:
            q = epsilon / k 

        loss += (q *  math.log(predictions[i]))

    return float(-loss)