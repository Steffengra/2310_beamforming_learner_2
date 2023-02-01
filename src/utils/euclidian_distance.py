
from numpy import (
    ndarray,
    sqrt,
)


def euclidian_distance(
        vec_1: ndarray,
        vec_2: ndarray,
) -> ndarray:
    """
    This function calculates the euclidean distance between two 3D vectors
    """

    x_dist = vec_1[0] - vec_2[0]
    y_dist = vec_1[1] - vec_2[1]
    z_dist = vec_1[2] - vec_2[2]

    dist = sqrt(x_dist**2 + y_dist**2 + z_dist**2)

    return dist
