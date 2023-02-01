
from numpy import (
    ndarray,
    arange,
    pi,
    exp,
    cos,
    arcsin,
)

from src.utils.spherical_to_cartesian_coordinates import (
    spherical_to_cartesian_coordinates,
)
from src.utils.euclidian_distance import (
    euclidian_distance,
)


class Satellite:

    def __init__(
            self,
            idx,
            spherical_coordinates: ndarray,
            antenna_nr: int,
            antenna_distance: float,
            wavelength: float,
    ) -> None:

        self.idx: int = idx
        self.spherical_coordinates: ndarray = spherical_coordinates
        self.cartesian_coordinates: ndarray = spherical_to_cartesian_coordinates(spherical_coordinates)

        self.antenna_nr: int = antenna_nr
        self.antenna_distance: float = antenna_distance  # antenna distance in meters
        self.wavelength: float = wavelength

        self.distance_to_users: dict = {}  # user_idx[int]: dist[float]
        self.aods_to_users: dict = {}  # user_idx[int]: aod[float]
        self.steering_vectors_to_users: dict = {}  # user_idx[int]: steering_vector[ndarray] \in 1 x antenna_nr

    def calculate_distance_to_users(
            self,
            users: list,
    ) -> None:
        # TODO: This doesn't change values of users that might have disappeared

        for user in users:
            self.distance_to_users[user.idx] = euclidian_distance(self.cartesian_coordinates,
                                                                  user.cartesian_coordinates)

    def calculate_aods_to_users(
            self,
            users: list
    ) -> None:
        """
        The calculation of the AODs is given by
        AOD = asin(
            ((orbit+radius_earth)^2 + sat_user_dist - radius_earth^2)
            /
            (2 * (orbit+radius_earth) * sat_user_dist)
        )
        """
        # TODO: Stimmt die Formel mit radius = coordinates?
        # TODO: Fehlt in Formel ein **2?
        # TODO: This doesn't change values of users that might have disappeared
        # TODO: Sind die in rad?
        for user in users:

            self.aods_to_users[user.idx] = arcsin(
                (
                    + self.spherical_coordinates[0]**2
                    + self.distance_to_users[user.idx]
                    - user.spherical_coordinates[0]**2
                )  # numerator
                /
                (
                    2 * self.spherical_coordinates[0] * self.distance_to_users[user.idx]
                )  # denominator
            )

    def calculate_steering_vectors(
            self,
            users: list,
    ) -> None:
        """
        This function provides the steering vectors for a given ULA and AOD
        """

        steering_idx = arange(0, self.antenna_nr) - (self.antenna_nr - 1) / 2

        for user in users:
            self.steering_vectors_to_users[user.idx] = exp(
                steering_idx * (
                    -1j * 2 * pi / self.wavelength
                    * self.antenna_distance
                    * cos(self.aods_to_users[user.idx])
                )
            )
