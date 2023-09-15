
import src


def update_sim(
    config: 'src.config.config.Config',
    satellite_manager: 'src.data.satellite_manager.SatelliteManager',
    user_manager: 'src.data.user_manager.UserManager',
) -> None:
    """Call all functions to update objects to the next simulation state."""

    user_manager.update_positions(config=config)
    satellite_manager.update_positions(config=config)

    satellite_manager.calculate_satellite_distances_to_users(users=user_manager.users)
    satellite_manager.calculate_satellite_aods_to_users(users=user_manager.users)
    satellite_manager.roll_estimation_errors()
    satellite_manager.update_channel_state_information(channel_model=config.channel_model,
                                                       users=user_manager.users)
    satellite_manager.update_erroneous_channel_state_information(channel_model=config.channel_model,
                                                                 users=user_manager.users)
