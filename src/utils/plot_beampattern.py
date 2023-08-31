
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import src
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.config.config import Config
from src.utils.update_sim import update_sim


def plot_beampattern(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        user_manager: 'src.data.user_manager.UserManager',
        w_precoder: np.ndarray,
        user_main_idx: int,
        angle_sweep_range: np.ndarray or None = None,
        plot_title: str or None = None,
) -> None:

    # 1: calculate beamformer for given positions
    # 2: update position of 1 user iteratively
    # 3: calculate new channel
    # 4: evaluate new channel + beamformer

    config_local = deepcopy(config)
    config_local.user_dist_bound = 0

    config.config_error_model.set_zero_error()

    satellite_manager_local = deepcopy(satellite_manager)
    user_manager_local = deepcopy(user_manager)

    update_sim(config=config_local, satellite_manager=satellite_manager_local, user_manager=user_manager_local)

    usr_main_pos_original = user_manager_local.users[user_main_idx].spherical_coordinates

    if angle_sweep_range is None:
        max_dist = (user_manager_local.users[-1].spherical_coordinates[2] - user_manager_local.users[0].spherical_coordinates[2])

        angle_sweep_range = np.arange(
            user_manager_local.users[0].spherical_coordinates[2] - 0.5 * max_dist,
            user_manager_local.users[-1].spherical_coordinates[2] + 0.5 * max_dist,
            (user_manager_local.users[-1].spherical_coordinates[2] - user_manager_local.users[0].spherical_coordinates[2]) / 100
        )

    sinrs = np.empty(len(angle_sweep_range))
    sum_power_gains = np.empty(len(angle_sweep_range))
    for angle_id, angle in enumerate(angle_sweep_range):
        user_manager_local.users[user_main_idx].update_position(
            [user_manager_local.users[user_main_idx].spherical_coordinates[0], user_manager_local.users[user_main_idx].spherical_coordinates[1], angle]
        )

        satellite_manager_local.update_positions(config=config_local)
        satellite_manager_local.calculate_satellite_distances_to_users(users=user_manager_local.users)
        satellite_manager_local.calculate_satellite_aods_to_users(users=user_manager_local.users)
        satellite_manager_local.roll_estimation_errors()
        satellite_manager_local.update_channel_state_information(channel_model=config_local.channel_model, users=user_manager_local.users)
        satellite_manager_local.update_erroneous_channel_state_information(channel_model=config.channel_model, users=user_manager_local.users)

        power_gain_user_main = abs(np.matmul(satellite_manager_local.channel_state_information[user_main_idx, :], w_precoder[:, user_main_idx])) ** 2
        power_fading_precoded_other_users_sigma_i = [
            abs(np.matmul(satellite_manager_local.channel_state_information[user_main_idx, :], w_precoder[:, other_user_idx])) ** 2
            for other_user_idx in range(len(user_manager_local.users)) if other_user_idx != user_main_idx
        ]
        sum_interference = sum(power_fading_precoded_other_users_sigma_i)

        power_gain_all_users = [
            abs(np.matmul(satellite_manager_local.channel_state_information[user_main_idx, :], w_precoder[:, other_user_idx])) ** 2
            for other_user_idx in range(len(user_manager_local.users))
        ]
        sum_power_gain = sum(power_gain_all_users)

        #sinr = power_gain_user_main / (config_local.noise_power_watt + sum_interference)
        sinr = power_gain_user_main
        sinrs[angle_id] = sinr
        sum_power_gains[angle_id] = sum_power_gain / config_local.noise_power_watt

    plt.plot(angle_sweep_range, sinrs, label='sinr user_main')
    plt.plot(angle_sweep_range, sum_power_gains, label='total receive power')
    plt.scatter(usr_main_pos_original[2], 0, label='user_main')
    other_user_angles = [user.spherical_coordinates[2] for user in user_manager_local.users if user.idx != user_main_idx]
    plt.scatter(other_user_angles, [0]*len(other_user_angles), label='user_other', color='green')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.xlabel('User main Position')
    # plt.ylabel('User main SINR')
    if plot_title is not None:
        plt.title(plot_title)

    plt.show()
