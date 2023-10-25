
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from sys import stdout

import numpy as np
from scipy import constants
from tensorflow import get_logger as tf_get_logger

from src.config.config_error_model import (
    ConfigErrorModel,
)
from src.config.config_sac_learner import (
    ConfigSACLearner,
)
from src.data.channel.los_channel_model import (
    los_channel_model,
)
from src.utils.get_wavelength import (
    get_wavelength,
)


class Config:
    """The config sets up all global parameters."""

    def __init__(
            self,
    ) -> None:

        self._pre_init()

        # General
        self.profile: bool = False  # performance profiling
        self.show_plots: bool = True

        self.verbosity: int = 1  # 0 = no prints, 1 = prints
        self._logging_level_stdio = logging.INFO  # DEBUG < INFO < WARNING < ERROR < CRITICAL < CRITICAL+1
        self._logging_level_file = logging.DEBUG
        self._logging_level_tensorflow = logging.WARNING
        self._logging_level_matplotlib = logging.INFO
        self.logfile_max_bytes: int = 10_000_000  # log file max size, one backup file is kept

        # Basic Communication Parameters
        self.freq: float = 2 * 10**9
        self.noise_power_watt: float = 10**(7 / 10) * 290 * constants.value('Boltzmann constant') * 30 * 10**6  # Noise power
        self.power_constraint_watt = 100  # in watt

        self.wavelength: float = get_wavelength(self.freq)

        # Orbit
        self.altitude_orbit: float = 600 * 10**3  # Orbit altitude d0
        self.radius_earth: float = 6378.1 * 10**3  # Earth radius RE, earth centered

        self.radius_orbit: float = self.altitude_orbit + self.radius_earth  # Orbit radius with Earth center r0, earth centered

        # User
        self.user_nr: int = 3  # Number of users
        self.user_gain_dBi: float = 0  # User gain in dBi
        self.user_dist_average: float = 100_000  # Average user distance in m
        self.user_dist_bound: float = 50_000  # Variance of user distance, uniform distribution [avg-bound, avg+bound]
        self.user_center_aod_earth_deg: float = 90  # Average center of users

        self.user_gain_linear: float = 10**(self.user_gain_dBi / 10)  # User gain linear

        # Satellite
        self.sat_nr: int = 1  # Number of satellites
        self.sat_tot_ant_nr: int = 10  # Total number of  Tx antennas, should be a number larger than sat nr
        self.sat_gain_dBi: float = 20  # Total sat TODO: Wert nochmal checken
        self.sat_dist_average: float = 10_000  # Average satellite distance in meter
        self.sat_dist_bound: float = 0  # Variance of sat distance, uniform distribution [avg-bound, avg+bound]
        self.sat_center_aod_earth_deg: float = 90  # Average center of satellites

        self.sat_gain_linear: float = 10**(self.sat_gain_dBi / 10)  # Gain per satellite linear
        self.sat_ant_nr: int = int(self.sat_tot_ant_nr / self.sat_nr)  # Number of Tx antennas per satellite
        self.sat_ant_gain_linear: float = self.sat_gain_linear / self.sat_tot_ant_nr  # Gain per satellite antenna
        self.sat_ant_dist: float = 3 * self.wavelength / 2  # Distance between antenna elements in meter

        # Channel Model
        self.channel_model = los_channel_model

        self._post_init()

    def _pre_init(
            self,
    ) -> None:

        self.rng = np.random.default_rng()
        self.logger = logging.getLogger()

        self.project_root_path = Path(__file__).parent.parent.parent
        self.performance_profile_path = Path(self.project_root_path, 'outputs', 'performance_profiles')
        self.output_metrics_path = Path(self.project_root_path, 'outputs', 'metrics')
        self.trained_models_path = Path(self.project_root_path, 'models')

        self.performance_profile_path.mkdir(parents=True, exist_ok=True)
        self.output_metrics_path.mkdir(parents=True, exist_ok=True)
        self.trained_models_path.mkdir(parents=True, exist_ok=True)

    def _post_init(
            self,
    ) -> None:

        # Error Model
        self.config_error_model = ConfigErrorModel(
            channel_model=self.channel_model,
            rng=self.rng,
            wavelength=self.wavelength,
            user_nr=self.user_nr,
        )

        # Learner
        self.config_learner = ConfigSACLearner(
            sat_nr=self.sat_nr,
            sat_ant_nr=self.sat_ant_nr,
            user_nr=self.user_nr,
        )

        # Logging
        self.logfile_path = Path(self.project_root_path, 'outputs', 'logs', 'log.txt')
        self.logfile_path.parent.mkdir(parents=True, exist_ok=True)
        self.__logging_setup()

        # Collected args
        self.satellite_args: dict = {
            'rng': self.rng,
            'antenna_nr': self.sat_ant_nr,
            'antenna_distance': self.sat_ant_dist,
            'antenna_gain_linear': self.sat_ant_gain_linear,
            'user_nr': self.user_nr,
            'freq': self.freq,
            'center_aod_earth_deg': self.sat_center_aod_earth_deg,
            'error_functions': self.config_error_model.error_rngs
        }

        self.user_args: dict = {
            'gain_linear': self.user_gain_linear,
        }

        self.mmse_args: dict = {
            'power_constraint_watt': self.power_constraint_watt,
            'noise_power_watt': self.noise_power_watt,
            'sat_nr': self.sat_nr,
            'sat_ant_nr': self.sat_ant_nr,
        }

        self.mrc_args: dict = {
            'power_constraint_watt': self.power_constraint_watt,
        }

        self.learned_precoder_args: dict = {
            'sat_nr': self.sat_nr,
            'sat_ant_nr': self.sat_ant_nr,
            'user_nr': self.user_nr,
            'power_constraint_watt': self.power_constraint_watt,
        }

    def __logging_setup(
            self,
    ) -> None:

        logging_formatter = logging.Formatter(
            '{asctime} : {levelname:8s} : {name:30} : {funcName:25} :: {message}',
            datefmt='%Y-%m-%d %H:%M:%S',
            style='{',
        )

        # Create Handlers
        logging_file_handler = RotatingFileHandler(self.logfile_path, maxBytes=self.logfile_max_bytes, backupCount=1)
        logging_stdio_handler = logging.StreamHandler(stdout)

        # Set Logging Level
        logging_file_handler.setLevel(self._logging_level_file)

        if self.verbosity == 0:
            logging_stdio_handler.setLevel(logging.CRITICAL + 1)
        else:
            logging_stdio_handler.setLevel(self._logging_level_stdio)

        tensorflow_logger = tf_get_logger()
        tensorflow_logger.setLevel(self._logging_level_tensorflow)
        if len(tensorflow_logger.handlers) > 0:
            tensorflow_logger.handlers.pop(0)  # remove tf handler, we've got our own

        matplotlib_logger = logging.getLogger('matplotlib')
        matplotlib_logger.setLevel(self._logging_level_matplotlib)

        self.logger.setLevel(logging.NOTSET)  # set primary logger level to lowest to catch all

        # Set Formatting
        logging_file_handler.setFormatter(logging_formatter)
        logging_stdio_handler.setFormatter(logging_formatter)

        # Add Handlers
        self.logger.addHandler(logging_file_handler)
        self.logger.addHandler(logging_stdio_handler)
