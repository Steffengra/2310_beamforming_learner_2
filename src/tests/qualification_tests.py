
import unittest
import numpy as np
from scipy import constants

import src

from src.config.config import Config
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.update_sim import update_sim

from src.data.precoder.mmse_precoder import mmse_precoder_normalized
from src.data.calc_sum_rate import calc_sum_rate
from src.data.channel.los_channel_model import (
    los_channel_model,
)
from src.utils.get_wavelength import (
    get_wavelength,
)


def get_precoder_mmse(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
) -> np.ndarray:
    w_mmse = mmse_precoder_normalized(
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        **config.mmse_args,
    )

    return w_mmse


class TestMMMSEPerformance(unittest.TestCase):

    def setUp(
            self,
    ) -> None:

        self.config = Config()
        self.config.show_plots = False
        self.config.verbosity = 0  # 0 = no prints, 1 = prints

    def test_no_error(
            self,
    ) -> None:

        # Basic Communication Parameters
        self.config.freq = 2 * 10**9
        self.config.noise_power_watt = 10**(7 / 10) * 290 * constants.value('Boltzmann constant') * 30 * 10**6  # Noise power
        self.config.power_constraint_watt = 100  # in watt

        self.config.wavelength = get_wavelength(self.config.freq)

        # Orbit
        self.config.altitude_orbit = 600 * 10**3  # Orbit altitude d0
        self.config.radius_earth = 6378.1 * 10**3  # Earth radius RE, earth centered

        self.config.radius_orbit = self.config.altitude_orbit + self.config.radius_earth  # Orbit radius with Earth center r0, earth centered

        # User
        self.config.user_nr = 3  # Number of users
        self.config.user_gain_dBi = 0  # User gain in dBi
        self.config.user_dist_average = 100_000  # Average user distance in m
        self.config.user_dist_bound = 0  # Variance of user distance, uniform distribution [avg-bound, avg+bound]
        self.config.user_center_aod_earth_deg = 90  # Average center of users

        self.config.user_gain_linear = 10**(self.config.user_gain_dBi / 10)  # User gain linear

        # Satellite
        self.config.sat_nr = 1  # Number of satellites
        self.config.sat_tot_ant_nr = 16  # Total number of  Tx antennas, should be a number larger than sat nr
        self.config.sat_gain_dBi = 20  # Total sat TODO: Wert nochmal checken
        self.config.sat_dist_average = 10_000  # Average satellite distance in meter
        self.config.sat_dist_bound = 0  # Variance of sat distance, uniform distribution [avg-bound, avg+bound]
        self.config.sat_center_aod_earth_deg = 90  # Average center of satellites

        self.config.sat_gain_linear = 10**(self.config.sat_gain_dBi / 10)  # Gain per satellite linear
        self.config.sat_ant_nr = int(self.config.sat_tot_ant_nr / self.config.sat_nr)  # Number of Tx antennas per satellite
        self.config.sat_ant_gain_linear = self.config.sat_gain_linear / self.config.sat_tot_ant_nr  # Gain per satellite antenna
        self.config.sat_ant_dist = 3 * self.config.wavelength / 2  # Distance between antenna elements in meter

        # Channel Model
        self.config.channel_model = los_channel_model

        self.config._post_init()

        self.config.config_error_model.set_zero_error()

        self.satellite_manager = SatelliteManager(config=self.config)
        self.user_manager = UserManager(config=self.config)

        update_sim(config=self.config, satellite_manager=self.satellite_manager, user_manager=self.user_manager)

        w_mmse = mmse_precoder_normalized(
            self.satellite_manager.erroneous_channel_state_information,
            **self.config.mmse_args,
        )

        sum_rate = calc_sum_rate(
            channel_state=self.satellite_manager.channel_state_information,
            w_precoder=w_mmse,
            noise_power_watt=self.config.noise_power_watt,
        )

        self.assertAlmostEqual(sum_rate, 4.957990385975026)  # checks 7 digits


if __name__ == '__main__':
    unittest.main()
