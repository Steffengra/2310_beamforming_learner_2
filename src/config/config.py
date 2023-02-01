
import logging
from numpy.random import (
    default_rng,
)
from scipy import (
    constants,
)

from src.data.channel.los_channel_model import (
    los_channel_model,
)
from src.config.config_error_model import (
    ConfigErrorModel,
)
from src.utils.get_wavelength import (
    get_wavelength,
)


class Config:
    """
    The config sets up all global parameters
    """

    def __init__(
            self,
    ) -> None:

        self._pre_init()

        # General
        self.profile: bool = True  # performance profiling

        # Basic Communication Parameters (freq, wavelength, noise, tx power)
        self.freq: float = 2 * 10**9
        self.noise_power_watt: float = 10**(7 / 10) * 290 * constants.value('Boltzmann constant') * 30 * 10**6  # Noise power TODO: is watt?
        self.power_constraint_watt = 100  # in watt

        self.wavelength: float = get_wavelength(self.freq)

        # Orbit
        self.altitude_orbit: float = 600 * 10**3  # Orbit altitude d0
        self.radius_earth: float = 6378.1 * 10**3  # Earth radius RE, earth centered

        self.radius_orbit: float = self.altitude_orbit + self.radius_earth  # Orbit radius with Earth center r0, earth centered

        # User
        self.user_nr: int = 5  # Number of users
        self.user_gain_dBi: float = 0  # User gain in dBi
        self.user_dist_average: float = 100**3  # Average user distance
        self.user_dist_variance: float = 0  # Variance of average user distance (normal distribution around the average user distance)
        self.user_center_aod_earth_deg: float = 90  # Average center of users

        self.user_gain_linear: float = 10**(self.user_gain_dBi / 10)  # User gain linear

        # Satellite
        self.sat_nr: int = 3  # Number of satellites
        self.sat_tot_ant_nr: int = 12  # Total number of  Tx antennas, should be a number larger than sat nr
        self.sat_gain_dBi: float = 20  # Total sat TODO: Wert nochmal checken
        self.sat_dist_average: float = 100**3  # Average satellite distance in meter
        self.sat_dist_variance: float = 0  # Variance of Average satellite distance (normal distribution)
        self.sat_center_aod_earth_deg: float = 90  # Average center of satellites

        self.sat_gain_linear: float = 10**(self.sat_gain_dBi / 10)  # Gain per satellite linear
        self.sat_ant_nr: int = int(self.sat_tot_ant_nr / self.sat_nr)  # Number of Tx antennas per satellite
        self.sat_ant_gain_linear: float = self.sat_gain_linear / self.sat_tot_ant_nr  # Gain per satellite antenna
        self.sat_ant_dist: float = self.wavelength / 2  # Distance between antenna elements in meter

        # Channel Model
        self.channel_model = los_channel_model

        # Sweep Settings
        self.monte_carlo_iterations: int = 1_000

        self._post_init()

    def _pre_init(
            self,
    ) -> None:
        self.rng = default_rng()
        self.logger = logging.getLogger()

    def _post_init(
            self,
    ) -> None:

        # Error Model
        self.error_model = ConfigErrorModel()

        # Collected args
        self.satellite_args: dict = {
            'rng': self.rng,
            'antenna_nr': self.sat_ant_nr,
            'antenna_distance': self.sat_ant_dist,
            'antenna_gain_linear': self.sat_ant_gain_linear,
            'freq': self.freq,
        }

        self.user_args: dict = {
            'gain_linear': self.user_gain_linear,
        }


if __name__ == '__main__':
    cfg = Config()
