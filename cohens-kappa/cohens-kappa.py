import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    rater1 = np.asarray(rater1)
    rater2 = np.asarray(rater2)

    if len(rater1) != len(rater2):
        raise ValueError("Both raters must have the same number of ratings.")

    # Get all unique categories
    categories = np.unique(np.concatenate((rater1, rater2)))
    n = len(rater1)

    # Confusion matrix
    confusion = np.zeros((len(categories), len(categories)), dtype=int)

    for a, b in zip(rater1, rater2):
        i = np.where(categories == a)[0][0]
        j = np.where(categories == b)[0][0]
        confusion[i, j] += 1

    # Observed agreement
    po = np.trace(confusion) / n

    # Expected agreement
    row_totals = confusion.sum(axis=1)
    col_totals = confusion.sum(axis=0)
    pe = np.sum(row_totals * col_totals) / (n * n)

    # Cohen's Kappa
    if pe == 1:
        return 1.0

    kappa = (po - pe) / (1 - pe)
    return kappa