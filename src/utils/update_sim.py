
from src.config.config import (
    Config,
)
from src.data.satellite_manager import (
    SatelliteManager,
)
from src.data.user_manager import (
    UserManager,
)


def update_sim(
    config: Config,
    satellite_manager: SatelliteManager,
    user_manager: UserManager,
) -> None:

    user_manager.update_positions(config=config)
    satellite_manager.update_positions(config=config)

    satellite_manager.calculate_satellite_distances_to_users(users=user_manager.users)
    satellite_manager.calculate_satellite_aods_to_users(users=user_manager.users)
    satellite_manager.calculate_steering_vectors_to_users(users=user_manager.users)
    satellite_manager.update_channel_state_information(channel_model=config.channel_model, users=user_manager.users)
    satellite_manager.update_erroneous_channel_state_information(error_model_config=config.error_model,
                                                                 users=user_manager.users)
