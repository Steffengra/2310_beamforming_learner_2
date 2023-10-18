
import numpy as np

from src.utils.spherical_to_cartesian_coordinates import (
    spherical_to_cartesian_coordinates,
)


class User:
    """
    A user object represents a physical user.
    """

    def __init__(
            self,
            idx: int,
            spherical_coordinates: np.ndarray,
            gain_linear: float,
    ) -> None:

        self.idx: int = idx
        self.spherical_coordinates: np.ndarray = spherical_coordinates
        self.cartesian_coordinates: np.ndarray = spherical_to_cartesian_coordinates(spherical_coordinates)

        self.gain_linear: float = gain_linear

    def update_position(
            self,
            spherical_coordinates: np.ndarray,
    ) -> None:
        """TODO: Comment"""

        self.spherical_coordinates = spherical_coordinates
        self.cartesian_coordinates = spherical_to_cartesian_coordinates(spherical_coordinates)
