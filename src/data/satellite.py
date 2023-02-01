
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

    def calculate_distance_to_users(
            self,
            users: list,
    ) -> None:
        # TODO: This doesnt change values of users that might have disappeared

        for user in users:
            self.distance_to_users[user.idx] = euclidian_distance(self.cartesian_coordinates,
                                                                  user.cartesian_coordinates)
