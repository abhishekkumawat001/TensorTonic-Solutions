def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    recommended= list(recommended)
    relevant= set(relevant)

    top_k = recommended[:k]

    hits = 0 
    for item in top_k:
        if item in relevant:
            hits += 1

    precision = hits / k if k > 0 else 0 

    recall = hits / len(relevant) if len(relevant) >0 else 0 
    return [precision , recall]