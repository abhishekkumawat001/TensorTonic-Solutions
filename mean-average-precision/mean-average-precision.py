import numpy as np


def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.

    Returns:
        (map_value, ap_per_query)
    """

    ap_per_query = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        # Sort by score in descending order
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]

        # Apply cutoff
        if k is not None:
            y_true_sorted = y_true_sorted[:k]

        # Total number of relevant items
        R = np.sum(y_true)

        # No relevant items
        if R == 0:
            ap_per_query.append(0.0)
            continue

        # Relevant items in ranked order
        relevant = (y_true_sorted == 1)

        # Number of relevant items seen up to each rank
        cumulative_relevant = np.cumsum(relevant)

        # Rank positions: 1, 2, ..., n
        ranks = np.arange(1, len(y_true_sorted) + 1)

        # Precision at every rank
        precision = cumulative_relevant / ranks

        # AP = sum of precision at relevant ranks / R
        ap = np.sum(precision * relevant) / R

        ap_per_query.append(ap)

    # Mean AP across queries
    map_value = np.mean(ap_per_query) if ap_per_query else 0.0

    return map_value, ap_per_query