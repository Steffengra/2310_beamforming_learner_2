
import unittest

from src.config.config import Config
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.update_sim import update_sim


class TestSystemState(unittest.TestCase):

    def setUp(
            self,
    ) -> None:

        self.config = Config()
        self.satellite_manager = SatelliteManager(config=self.config)
        self.user_manager = UserManager(config=self.config)

        self.user_manager.update_positions(config=self.config)
        self.satellite_manager.update_positions(config=self.config)

        update_sim(config=self.config, satellite_manager=self.satellite_manager, user_manager=self.user_manager)


if __name__ == '__main__':
    unittest.main()
