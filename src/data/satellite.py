
from numpy import (
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
            spherical_coordinates,
    ) -> None:

        self.idx: int = idx
        self.spherical_coordinates = spherical_coordinates
        self.cartesian_coordinates = spherical_to_cartesian_coordinates(spherical_coordinates)

        self.distance_to_users: dict = {}  # user_idx[int]: dist[float]
        self.aods_to_users: dict = {}  # user_idx[int]: aod[float]

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
                )
                /
                (
                    2 * self.spherical_coordinates[0] * self.distance_to_users[user.idx]
                )
            )
