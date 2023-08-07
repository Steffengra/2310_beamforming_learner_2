
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


if __name__ == '__main__':
    unittest.main()
