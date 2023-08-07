
import unittest

from src.config.config import Config
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.models.helpers.get_state import (
    get_state_erroneous_channel_state_information,
    get_state_aods,
)


class TestSystemState(unittest.TestCase):

    def setUp(
            self,
    ) -> None:

        self.config = Config()
        self.satellite_manager = SatelliteManager(config=self.config)
        self.user_manager = UserManager(config=self.config)

        self.user_manager.update_positions(config=self.config)
        self.satellite_manager.update_positions(config=self.config)

        self.satellite_manager.calculate_satellite_distances_to_users(users=self.user_manager.users)
        self.satellite_manager.calculate_satellite_aods_to_users(users=self.user_manager.users)
        self.satellite_manager.calculate_steering_vectors_to_users(users=self.user_manager.users)
        self.satellite_manager.update_channel_state_information(channel_model=self.config.channel_model,
                                                                users=self.user_manager.users)
        self.satellite_manager.update_erroneous_channel_state_information(error_model_config=self.config.error_model,
                                                                          users=self.user_manager.users)

    def test_state_erroneous_csi_rad_phase_normalization(
            self,
    ):
        csi = get_state_erroneous_channel_state_information(
            satellite_manager=self.satellite_manager,
            csi_format='rad_phase',
            norm_state=True,
        )
        self.assertTrue(all(abs(csi) < 5))

    def test_state_erroneous_csi_real_imag_normalization(
            self,
    ):
        csi = get_state_erroneous_channel_state_information(
            satellite_manager=self.satellite_manager,
            csi_format='real_imag',
            norm_state=True,
        )
        self.assertTrue(all(abs(csi) < 5))

    def test_state_aods_normalization(
            self,
    ):
        aods = get_state_aods(
            satellite_manager=self.satellite_manager,
            norm_state=True
        )
        self.assertTrue(all(abs(aods) < 5))


if __name__ == '__main__':
    unittest.main()
