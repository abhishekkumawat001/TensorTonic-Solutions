def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    hits = 0
    n_users = len(recommendations)

    for recs, truth in zip(recommendations, ground_truth):
        top_k = set(recs[:k])
        relevant = set(truth)

        if top_k & relevant:   # Non-empty intersection
            hits += 1

    return hits / n_users