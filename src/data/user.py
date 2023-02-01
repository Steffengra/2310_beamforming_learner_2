
from src.utils.spherical_to_cartesian_coordinates import (
    spherical_to_cartesian_coordinates,
)


class User:

    def __init__(
            self,
            idx,
            spherical_coordinates,
            gain_linear,
    ) -> None:

        self.idx: int = idx
        self.spherical_coordinates = spherical_coordinates
        self.cartesian_coordinates = spherical_to_cartesian_coordinates(spherical_coordinates)

        self.gain_linear: float = gain_linear
