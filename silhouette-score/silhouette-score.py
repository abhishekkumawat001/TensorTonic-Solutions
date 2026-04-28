import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # Write code here
    X = np.asarray(X)
    labels = np.asarray(labels)

    n = len(X)
    unique_labels = np.unique(labels)
    silhouette_score = []

    for i in range(n):
        same_cluster = labels == labels[i]
        same_cluster[i] = False # exclude itself

        # a(i): average intra cluster distance
        if np.sum(same_cluster) > 0:
            a = np.mean(np.linalg.norm(X[i] - X[same_cluster], axis=1))
        else:
            a = 0.0

        # b(i) minimum average distance to points in the other cluster 
        b = float("inf")

        for label in unique_labels:
            if label == labels[i]:
                continue
            other_cluster = labels == label
            if np.sum(other_cluster) > 0:
                dist = np.mean(np.linalg.norm(X[i] - X[other_cluster], axis = 1))
                b = min(b , dist)

        if max(a , b) == 0:
            s = 0.0
        else:
            s = (b-a)/ max(a , b)

        silhouette_score.append(s)

    return float(np.mean(silhouette_score))
            