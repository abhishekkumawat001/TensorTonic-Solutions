def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    y_true = list(y_true)
    y_pred = list(y_pred)
    tp = 0
    fp = 0 
    fn = 0
    
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            tp += 1
        else:
            fp += 1
            fn += 1
    denominator = (2 * tp + fp + fn)

    if denominator == 0:
        return 0.0

    return (2 * tp) / denominator        