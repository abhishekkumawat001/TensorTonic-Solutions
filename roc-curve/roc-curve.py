import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Sort by descending score, and for ties ensure stable grouping
    order = np.lexsort((1 - y_true, -y_score))  # secondary key helps grouping
    y_true = y_true[order]
    y_score = y_score[order]

    # Total positives and negatives
    P = np.sum(y_true)
    N = len(y_true) - P

    # Cumulative sums of TP and FP
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)

    # Find indices where score changes (unique thresholds)
    distinct = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct, len(y_score) - 1]

    # Select TP/FP at threshold points
    tps = tps[threshold_idxs]
    fps = fps[threshold_idxs]
    thresholds = y_score[threshold_idxs]

    # Convert to rates
    tpr = tps / P if P > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / N if N > 0 else np.zeros_like(fps, dtype=float)

    # Add starting point (0,0) with threshold = inf
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, thresholds]

    # Ensure ending point (1,1)
    if fpr[-1] != 1.0 or tpr[-1] != 1.0:
        tpr = np.r_[tpr, 1.0]
        fpr = np.r_[fpr, 1.0]
        thresholds = np.r_[thresholds, thresholds[-1]]

    return fpr, tpr, thresholds