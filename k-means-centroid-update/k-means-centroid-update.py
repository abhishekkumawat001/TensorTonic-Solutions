import numpy as np 

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    points = np.asarray(points, dtype=float)
    assignments = np.asarray(assignments)

    # Number of dimensions
    n_features = points.shape[1]

    # Initialize centroids
    centroids = np.zeros((k, n_features), dtype=float)

    # Compute mean for each cluster
    for j in range(k):
        cluster_points = points[assignments == j]

        if len(cluster_points) > 0:
            centroids[j] = np.mean(cluster_points, axis=0)

    return centroids.tolist()