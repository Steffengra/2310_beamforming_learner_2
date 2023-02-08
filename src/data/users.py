
from numpy import (
    ndarray,
    array,
    arange,
    zeros,
    ones,
    sqrt,
    arccos,
    pi,
)

from src.config.config import (
    Config,
)
from src.data.user import (
    User,
)


class Users:
    """
    Users holds all user objects
    """

    def __init__(
            self,
            config: Config,
    ) -> None:

        self.rng = config.rng
        self.logger = config.logger.getChild(__name__)

        self.users: list = []
        self._initialize_users(config=config)

        self.logger.info('user setup complete')

    def calc_spherical_coordinates(
            self,
            config,
    ) -> (ndarray, list):

        # calculate average user positions
        user_pos_average = (arange(0, config.user_nr) - (config.user_nr - 1) / 2) * config.user_dist_average

        # add random value on user distances
        user_dist = user_pos_average + self.rng.normal(loc=0, scale=sqrt(config.user_dist_variance), size=config.user_nr)

        # calculate user_aods_diff_earth_rad
        user_aods_diff_earth_rad = zeros(config.user_nr)

        for user_idx in range(config.user_nr):

            if user_dist[user_idx] < 0:
                user_aods_diff_earth_rad[user_idx] = -1 * arccos(1 - 0.5 * (user_dist[user_idx] / config.radius_earth)**2)
            elif user_dist[user_idx] >= 0:
                user_aods_diff_earth_rad[user_idx] = arccos(1 - 0.5 * (user_dist[user_idx] / config.radius_earth)**2)

        user_center_aod_earth_rad = config.user_center_aod_earth_deg * pi / 180

        # TODO: if any(user_pos_average == 0) == 1, vllt Fallunterscheidung für gerade und ungerade

        # calculate user_aods_earth_rad
        user_aods_earth_rad = user_center_aod_earth_rad + user_aods_diff_earth_rad

        # create user objects
        user_radii = config.radius_earth * ones(config.user_nr)
        user_inclinations = pi / 2 * ones(config.user_nr)

        user_spherical_coordinates = array([user_radii, user_inclinations, user_aods_earth_rad])

        return user_spherical_coordinates

    def _initialize_users(
            self,
            config: Config,
    ) -> None:

        user_spherical_coordinates = self.calc_spherical_coordinates(config=config)

        for user_idx in range(config.user_nr):
            self.users.append(
                User(
                    idx=user_idx,
                    spherical_coordinates=user_spherical_coordinates[:, user_idx],
                    **config.user_args,
                )
            )

    def update_positions(
            self,
            config,
    ) -> None:

        user_spherical_coordinates = self.calc_spherical_coordinates(config=config)

        for user in self.users:
            user.update_position(
                spherical_coordinates=user_spherical_coordinates[:, user.idx],
            )
