import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """

    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)

    single = (points.ndim == 1)

    if single:
        points = points[np.newaxis, :]   # (1, 3)

    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))  # (N, 4)

    # Apply transformation
    transformed_h = points_h @ T.T        # (N, 4)

    # Convert back to Cartesian coordinates
    transformed = transformed_h[:, :3]

    return transformed[0] if single else transformed