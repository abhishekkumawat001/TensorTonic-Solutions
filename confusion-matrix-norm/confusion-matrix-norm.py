import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.

    Rows = true labels
    Columns = predicted labels
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if normalize not in ('none', 'true', 'pred', 'all'):
        raise ValueError("normalize must be 'none', 'true', 'pred', or 'all'")

    # Infer number of classes
    if num_classes is None:
        if y_true.size == 0 and y_pred.size == 0:
            num_classes = 0
        else:
            num_classes = int(max(y_true.max(), y_pred.max())) + 1

    # Raw confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    # No normalization
    if normalize == 'none':
        return cm

    cm = cm.astype(float)

    if normalize == 'true':
        # Normalize each row
        row_sums = cm.sum(axis=1, keepdims=True)
        np.divide(cm, row_sums, out=cm, where=row_sums != 0)

    elif normalize == 'pred':
        # Normalize each column
        col_sums = cm.sum(axis=0, keepdims=True)
        np.divide(cm, col_sums, out=cm, where=col_sums != 0)

    elif normalize == 'all':
        # Normalize by total number of samples
        total = cm.sum()
        if total != 0:
            cm /= total

    return cm