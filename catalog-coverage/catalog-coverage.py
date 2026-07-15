def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.

    Parameters:
        recommendations: List[List[int]]
            Recommendation list for each user.
        n_items: int
            Total number of items in the catalog.

    Returns:
        float: Catalog coverage.
    """
    unique_items = set()

    for rec_list in recommendations:
        unique_items.update(rec_list)

    return len(unique_items) / n_items