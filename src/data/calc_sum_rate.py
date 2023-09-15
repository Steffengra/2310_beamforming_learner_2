
import numpy as np


def calc_sum_rate(
    channel_state: np.ndarray,
    w_precoder: np.ndarray,
    noise_power_watt: float
) -> float:
    """TODO: comment"""

    user_nr = channel_state.shape[0]

    sinr_users = np.zeros(user_nr)

    for user_idx in range(user_nr):
        channel_user_H_k = channel_state[user_idx, :]
        power_fading_precoded_sigma_x = abs(np.matmul(channel_user_H_k, w_precoder[:, user_idx]))**2

        power_fading_precoded_other_users_sigma_i = [
            abs(np.matmul(channel_user_H_k, w_precoder[:, other_user_idx]))**2
            for other_user_idx in range(user_nr) if other_user_idx != user_idx
        ]

        sum_power_fading_precoded_other_users_sigma_int = sum(power_fading_precoded_other_users_sigma_i)

        sinr_users[user_idx] = (
                power_fading_precoded_sigma_x / (noise_power_watt + sum_power_fading_precoded_other_users_sigma_int)
        )

    info_rate_users = np.log2(1 + sinr_users)
    sum_rate = sum(info_rate_users)

    return sum_rate
