import numpy as np 

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    points = np.asarray(points, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    # Compute squared Euclidean distances
    # points[:, None, :]      -> (N, 1, D)
    # centroids[None, :, :]   -> (1, K, D)
    # result                  -> (N, K)
    distances = np.sum(
        (points[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2
    )

    # Index of nearest centroid for each point
    assignments = np.argmin(distances, axis=1)

    return assignments.tolist()