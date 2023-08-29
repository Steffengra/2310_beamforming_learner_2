
import numpy as np


def euclidian_distance(
        vec_1: np.ndarray,
        vec_2: np.ndarray,
) -> np.ndarray:
    """
    This function calculates the Euclidean distance between two 3D vectors
    """

    x_dist = vec_1[0] - vec_2[0]
    y_dist = vec_1[1] - vec_2[1]
    z_dist = vec_1[2] - vec_2[2]

    dist = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)

    return dist
