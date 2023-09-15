
import numpy as np


def calc_sum_rate_no_iui(
    channel_state: np.ndarray,
    w_precoder: np.ndarray,
    noise_power_watt: float,
) -> float:
    """TODO: Comment"""

    user_nr = channel_state.shape[0]

    sinr_users = np.zeros(user_nr)

    for user_id in range(user_nr):

        H_k = channel_state[user_id, :]

        sigma_x = abs(np.matmul(H_k, w_precoder[:, user_id]))**2
        sigma_int = 0

        sinr_users[user_id] = sigma_x / (noise_power_watt + sigma_int)

    info_rate = np.zeros(user_nr)

    for user_id in range(user_nr):

        info_rate[user_id] = np.log2(1 + sinr_users[user_id])

    sum_rate_without_iui = 1 / user_nr * sum(info_rate)

    return sum_rate_without_iui
