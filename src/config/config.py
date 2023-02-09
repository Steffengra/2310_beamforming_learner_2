
import logging
from pathlib import (
    Path,
)
from sys import (
    stdout,
)
from numpy.random import (
    default_rng,
)
from scipy import (
    constants,
)
from tensorflow import (
    get_logger as tf_get_logger,
)

from src.config.config_error_model import (
    ConfigErrorModel,
)
from src.config.config_td3_learner import (
    ConfigTD3Learner,
)
from src.data.channel.los_channel_model import (
    los_channel_model,
)
from src.utils.get_wavelength import (
    get_wavelength,
)


# TODO: At the moment, the policy network output is softmax. thats not necessary, but if
#  we change it, we also need to change the scale of the exploration noise somehow
# TODO: Implement copy config to saved model
# TODO: improve calc sum rate performance
# TODO: Fix add random distribution


class Config:
    """
    The config sets up all global parameters
    """

    def __init__(
            self,
    ) -> None:

        self._pre_init()

        # General
        self.profile: bool = False  # performance profiling
        self.show_plots: bool = True

        self.verbosity: int = 1  # 0 = no prints, 1 = prints
        self._logging_level_stdio = logging.INFO  # DEBUG < INFO < WARNING < ERROR < CRITICAL
        self._logging_level_file = logging.WARNING
        self._logging_level_tensorflow = logging.WARNING

        # Basic Communication Parameters (freq, wavelength, noise, tx power)
        self.freq: float = 2 * 10**9
        self.noise_power_watt: float = 10**(7 / 10) * 290 * constants.value('Boltzmann constant') * 30 * 10**6  # Noise power
        self.power_constraint_watt = 100  # in watt

        self.wavelength: float = get_wavelength(self.freq)

        # Orbit
        self.altitude_orbit: float = 600 * 10**3  # Orbit altitude d0
        self.radius_earth: float = 6378.1 * 10**3  # Earth radius RE, earth centered

        self.radius_orbit: float = self.altitude_orbit + self.radius_earth  # Orbit radius with Earth center r0, earth centered

        # User
        self.user_nr: int = 4  # Number of users
        self.user_gain_dBi: float = 0  # User gain in dBi
        self.user_dist_average: float = 10_000  # Average user distance in m
        self.user_dist_variance: float = 0  # Variance of average user distance (normal distribution around the average user distance)
        self.user_center_aod_earth_deg: float = 90  # Average center of users

        self.user_gain_linear: float = 10**(self.user_gain_dBi / 10)  # User gain linear

        # Satellite
        self.sat_nr: int = 3  # Number of satellites
        self.sat_tot_ant_nr: int = 12  # Total number of  Tx antennas, should be a number larger than sat nr
        self.sat_gain_dBi: float = 20  # Total sat TODO: Wert nochmal checken
        self.sat_dist_average: float = 100_000  # Average satellite distance in meter
        self.sat_dist_variance: float = 0  # Variance of Average satellite distance (normal distribution)
        self.sat_center_aod_earth_deg: float = 90  # Average center of satellites

        self.sat_gain_linear: float = 10**(self.sat_gain_dBi / 10)  # Gain per satellite linear
        self.sat_ant_nr: int = int(self.sat_tot_ant_nr / self.sat_nr)  # Number of Tx antennas per satellite
        self.sat_ant_gain_linear: float = self.sat_gain_linear / self.sat_tot_ant_nr  # Gain per satellite antenna
        self.sat_ant_dist: float = self.wavelength / 2  # Distance between antenna elements in meter

        # Channel Model
        self.channel_model = los_channel_model

        # Learner
        # self.config_learner = ConfigTD3Learner(size_state=self.sat_nr*self.user_nr,
        #                                        num_actions=2*self.sat_nr*self.sat_ant_nr*self.user_nr)
        self.config_learner = ConfigSACLearner(
            # size_state=self.sat_nr*self.user_nr,
            size_state=2*self.sat_nr*self.sat_ant_nr*self.user_nr,
            num_actions=2*self.sat_nr*self.sat_ant_nr*self.user_nr,
        )

        self._post_init()

    def _pre_init(
            self,
    ) -> None:
        self.rng = default_rng()
        self.logger = logging.getLogger()

        self.project_root_path = Path(__file__).parent.parent.parent
        self.performance_profile_path = Path(self.project_root_path, 'outputs', 'performance_profiles')
        self.trained_models_path = Path(self.project_root_path, 'models')

        self.performance_profile_path.mkdir(parents=True, exist_ok=True)
        self.trained_models_path.mkdir(parents=True, exist_ok=True)

    def _post_init(
            self,
    ) -> None:

        # Error Model
        self.error_model = ConfigErrorModel()

        # Logging
        self.logfile_path = Path(self.project_root_path, 'outputs', 'logs', 'log.txt')
        self.__logging_setup()

        # Collected args
        self.satellite_args: dict = {
            'rng': self.rng,
            'antenna_nr': self.sat_ant_nr,
            'antenna_distance': self.sat_ant_dist,
            'antenna_gain_linear': self.sat_ant_gain_linear,
            'freq': self.freq,
            'center_aod_earth_deg': self.sat_center_aod_earth_deg,
        }

        self.user_args: dict = {
            'gain_linear': self.user_gain_linear,
        }

    def __logging_setup(
            self,
    ) -> None:
        logging_formatter = logging.Formatter(
            '{asctime} : {levelname:8s} : {name:30} : {funcName:20s} :: {message}',
            datefmt='%Y-%m-%d %H:%M:%S',
            style='{',
        )

        # Create Handlers
        logging_file_handler = logging.FileHandler(self.logfile_path)
        logging_stdio_handler = logging.StreamHandler(stdout)

        # Set Logging Level
        logging_file_handler.setLevel(self._logging_level_file)
        logging_stdio_handler.setLevel(self._logging_level_stdio)

        tensorflow_logger = tf_get_logger()
        tensorflow_logger.setLevel(self._logging_level_tensorflow)

        self.logger.setLevel(logging.NOTSET)  # set primary logger level to lowest to catch all

        # Set Formatting
        logging_file_handler.setFormatter(logging_formatter)
        logging_stdio_handler.setFormatter(logging_formatter)

        # Add Handlers
        self.logger.addHandler(logging_file_handler)
        self.logger.addHandler(logging_stdio_handler)

        # Check Log File Size
        large_log_file_size = 30_000_000
        if self.logfile_path.stat().st_size > large_log_file_size:
            self.logger.warning(f'log file size >{large_log_file_size / 1_000_000} MB')
