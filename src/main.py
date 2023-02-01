
from numpy import (
    arange,
)

from src.config.config import (
    Config,
)
from src.data.satellites import (
    Satellites,
)
from src.data.users import (
    Users,
)
from src.data.los_channel_model import (
    los_channel_model,
)


def main():

    config = Config()
    satellites = Satellites(config=config)
    users = Users(config=config)
    CSIT_error = arange(0, 1, 0.1)

    satellites.calculate_satellite_distances_to_users(users=users.users)
    satellites.calculate_satellite_aods_to_users(users=users.users)
    satellites.calculate_steering_vectors_to_users(users=users.users)

    satellites.update_channel_state_information(channel_model=los_channel_model, users=users.users)
    satellites.update_erroneous_channel_state_information(error_model_config=config.error_model, users=users.users)


if __name__ == '__main__':
    main()
