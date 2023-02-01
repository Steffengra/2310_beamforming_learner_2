
from numpy import (
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
    Satellites holds all satellite objects
    TODO: And helper functions?
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

        self.satellites = []
        for sat_idx in range(config.sat_nr):
            self.satellites.append(
                Satellite(
                    idx=sat_idx,
                    spherical_coordinates=(sat_radii[sat_idx],
                                           sat_inclinations[sat_idx],
                                           sat_aods_earth_rad[sat_idx])
                )
            )

        # TODO: ALL?
