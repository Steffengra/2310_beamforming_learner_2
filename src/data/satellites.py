
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

        # calculate average satellite positions
        sat_pos_average = (arange(0, config.sat_nr) - (config.sat_nr - 1) / 2) * config.sat_dist_average

        # add random value on satellite distances
        sat_dist = sat_pos_average + sqrt(config.sat_dist_average) * self.rng.normal(size=config.sat_nr)

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

        self.satellites = []
        for sat_idx in range(config.sat_nr):
            self.satellites.append(
                Satellite(
                    idx=sat_idx,
                    spherical_coordinates=sat_spherical_coordinates[:, sat_idx],
                    **config.satellite_args,
                )
            )

        self.channel_state_information: ndarray = array([])  # TODO ndarray[sat_idx, ant_idx, user_idx]

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
        # TODO: Realistische Annahme, wavelength wird beim satellite hinterlegt?
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

        for satellite in self.satellites:
            satellite.update_channel_state_information(channel_model=channel_model, users=users)

        channel_state_per_satellite = []
        for satellite in self.satellites:
            channel_state_per_satellite.append(satellite.channel_state_to_users)
        self.channel_state_information = array(channel_state_per_satellite)
        # TODO: this flips the indices to self.csit[sat_idx, ant_idx, user_idx],
        #  imo makes more sense, but is that ok with alea?
