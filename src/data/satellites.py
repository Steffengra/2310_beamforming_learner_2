
from numpy import (
    ndarray,
    array,
    reshape,
    clip,
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
from src.data.satellite import (
    Satellite,
)


class Satellites:
    """
    Satellites holds all satellite objects and helper functions
    """

    def __init__(
            self,
            config: Config,
    ) -> None:

        self.rng = config.rng
        self.logger = config.logger.getChild(__name__)

        self.satellites: list[Satellite] = []
        self._initialize_satellites(config=config)

        self.channel_state_information: ndarray = array([])  # ndarray \in dim_user x (nr_antennas * nr_satellites)
                                                             #  per user: sat 1 ant1, sat 1 ant 2, sat 1 ant 3, sat 2 ant 1, ...
        self.erroneous_channel_state_information: ndarray = array([])  # ndarray \in dim_user x (nr_antennas * nr_satellites)

        self.logger.info('satellites setup complete')

    def calc_spherical_coordinates(
            self,
            config,
    ) -> ndarray:

        # calculate average satellite positions
        sat_pos_average = (arange(0, config.sat_nr) - (config.sat_nr - 1) / 2) * config.sat_dist_average

        # add random value on satellite distances
        random_factor = clip(self.rng.normal(loc=0, scale=sqrt(config.sat_dist_variance), size=config.sat_nr),
                             a_min=-config.sat_dist_average/2+1,
                             a_max=config.sat_dist_average/2-1)
        sat_dist = sat_pos_average + random_factor

        # calculate sat_aods_diff_earth_rad
        sat_aods_diff_earth_rad = zeros(config.sat_nr)

        for sat_idx in range(config.sat_nr):

            if sat_dist[sat_idx] < 0:
                sat_aods_diff_earth_rad[sat_idx] = -1 * arccos(1 - 0.5 * (sat_dist[sat_idx] / config.radius_orbit)**2)
            elif sat_dist[sat_idx] >= 0:
                sat_aods_diff_earth_rad[sat_idx] = arccos(1 - 0.5 * (sat_dist[sat_idx] / config.radius_orbit)**2)

        # calculate sat_center_aod_earth_rad
        sat_center_aod_earth_rad = config.sat_center_aod_earth_deg * pi / 180

        # TODO: if any(sat_pos_average == 0) == 1, vllt Fallunterscheidung für gerade und ungerade

        # calculate sat_aods_earth_rad
        sat_aods_earth_rad = sat_center_aod_earth_rad + sat_aods_diff_earth_rad

        # create satellite objects
        sat_radii = config.radius_orbit * ones(config.sat_nr)
        sat_inclinations = pi / 2 * ones(config.sat_nr)

        sat_spherical_coordinates = array([sat_radii, sat_inclinations, sat_aods_earth_rad])

        return sat_spherical_coordinates

    def _initialize_satellites(
            self,
            config: Config,
    ) -> None:
        """
        Initializes satellite object list for given configuration
        """

        sat_spherical_coordinates = self.calc_spherical_coordinates(config=config)

        for sat_idx in range(config.sat_nr):
            self.satellites.append(
                Satellite(
                    idx=sat_idx,
                    spherical_coordinates=sat_spherical_coordinates[:, sat_idx],
                    **config.satellite_args,
                )
            )

    def update_positions(
            self,
            config,
    ) -> None:

        sat_spherical_coordinates = self.calc_spherical_coordinates(config=config)

        for satellite in self.satellites:
            satellite.update_position(
                spherical_coordinates=sat_spherical_coordinates[:, satellite.idx],
            )

    def calculate_satellite_distances_to_users(
            self,
            users: list,
    ) -> None:
        """
        This function calculates the distances between each satellite and user
        """

        for satellite in self.satellites:
            satellite.calculate_distance_to_users(users=users)

    def calculate_satellite_aods_to_users(
            self,
            users: list,
    ) -> None:
        """
        This function calculates the AODs (angles of departure) from each satellite to
        each user (Earth and satellite orbits are assumed to be circular)
        """

        for satellite in self.satellites:
            satellite.calculate_aods_to_users(users=users)

    def calculate_steering_vectors_to_users(
            self,
            users: list,
    ) -> None:
        """
        This function calculates the steering vectors (one value per antenna) for each satellite to
        each user
        """

        for satellite in self.satellites:
            satellite.calculate_steering_vectors(users=users)

    def update_channel_state_information(
            self,
            channel_model,
            users: list,
    ) -> None:
        """
        This function builds channel state information between each satellite antenna and user, then
        accumulates all into a global channel state information matrix
        """

        # TODO: This will break when satellites dont have the same nr of antennas
        # TODO: This will also produce weird results when users or sats are not numbered consecutively

        # update channel state per satellite
        for satellite in self.satellites:
            satellite.update_channel_state_information(channel_model=channel_model, users=users)

        # build global channel state information
        channel_state_per_satellite = zeros((len(users), self.satellites[0].antenna_nr, len(self.satellites)),
                                            dtype='complex')
        for satellite in self.satellites:
            channel_state_per_satellite[:, :, satellite.idx] = satellite.channel_state_to_users
        self.channel_state_information = reshape(
            channel_state_per_satellite, (len(users), self.satellites[0].antenna_nr * len(self.satellites)))

    def update_erroneous_channel_state_information(
            self,
            error_model_config,
            users: list,
    ) -> None:

        # TODO: This will break when satellites dont have the same nr of antennas
        # TODO: This will also produce weird results when users or sats are not numbered consecutively

        # apply error model per satellite
        for satellite in self.satellites:
            satellite.update_erroneous_channel_state_information(error_model_config=error_model_config, users=users)

        # gather global erroneous channel state information
        erroneous_channel_state_per_satellite = zeros(
            (len(users), self.satellites[0].antenna_nr, len(self.satellites)),
            dtype='complex',
        )
        for satellite in self.satellites:
            erroneous_channel_state_per_satellite[:, :, satellite.idx] = satellite.erroneous_channel_state_to_users
        self.erroneous_channel_state_information = reshape(
            erroneous_channel_state_per_satellite, (len(users), self.satellites[0].antenna_nr * len(self.satellites)))

    def get_aods_to_users(
            self,
    ) -> ndarray:

        aods_to_users = zeros((len(self.satellites), len(self.satellites[0].aods_to_users)))
        for satellite_id, satellite in enumerate(self.satellites):
            aods_to_users[satellite_id, :] = array(list(satellite.aods_to_users.values()))

        return aods_to_users
