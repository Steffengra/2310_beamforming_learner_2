
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors

import src
from src.data.channel.get_steering_vec import get_steering_vec
from src.config.config_plotting import generic_styling


def plot_beampattern(
        satellite: 'src.data.satellite.Satellite',
        users: list['src.data.user.User'],
        w_precoder: np.ndarray,
        angle_sweep_range: np.ndarray or None = None,
        plot_title: str or None = None,
) -> None:
    """Plots beam power toward each user from the point of view of a satellite for a given precoding w_precoder."""

    # create a figure
    fig, ax = plt.subplots()

    # mark user positions
    for user_idx in range(len(users)):
        ax.scatter(
            satellite.aods_to_users[user_idx],
            0,
            label=f'user {user_idx}'
        )
        ax.axvline(
            satellite.aods_to_users[user_idx],
            color=mpl_colors.TABLEAU_COLORS[list(mpl_colors.TABLEAU_COLORS.keys())[user_idx]],
            linestyle='dashed'
        )

    # calculate auto x axis scaling
    if angle_sweep_range is None:
        max_dist = users[-1].spherical_coordinates[2] - users[0].spherical_coordinates[2]

        angle_sweep_range = np.arange(
            users[0].spherical_coordinates[2] - 0.5 * max_dist,
            users[-1].spherical_coordinates[2] + 0.5 * max_dist,
            (users[-1].spherical_coordinates[2] - users[0].spherical_coordinates[2]) / 100
        )

    # sweep power gains for each user depending on their angle
    power_gains = np.empty((len(users), len(angle_sweep_range)))
    for user_idx in range(len(users)):

        # sweep angles
        for angle_id, angle in enumerate(angle_sweep_range):

            steering_vector_to_user = get_steering_vec(satellite=satellite, phase_aod_steering=np.cos(angle))

            power_gain_user_main = abs(np.matmul(steering_vector_to_user, w_precoder[:, user_idx])) ** 2

            power_gains[user_idx, angle_id] = power_gain_user_main

        ax.plot(angle_sweep_range, power_gains[user_idx, :])

    ax.legend()
    ax.set_xlabel('User AOD from satellite')
    ax.set_ylabel('Power Gain')
    if plot_title is not None:
        ax.set_title(plot_title)

    generic_styling(ax=ax)
