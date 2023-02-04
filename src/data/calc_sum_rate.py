
from numpy import (
    matmul,
    zeros,
    log2,
    sum as np_sum,
)


def calc_sum_rate(
        channel_state,
        w_precoder,
        noise_power_watt
) -> float:
    """
    TODO: comment
    """

    user_nr = channel_state.shape[0]

    sinr_users = zeros(user_nr)

    for user_idx in range(user_nr):
        channel_user_H_k = channel_state[user_idx, :]
        power_fading_precoded_sigma_x = abs(matmul(channel_user_H_k, w_precoder[:, user_idx]))**2

        power_fading_precoded_other_users_sigma_i = [
            abs(matmul(channel_user_H_k, w_precoder[:, other_user_idx]))**2
            for other_user_idx in range(user_nr) if other_user_idx != user_idx
        ]

        sum_power_fading_precoded_other_users_sigma_int = sum(power_fading_precoded_other_users_sigma_i)

        sinr_users[user_idx] = (
                power_fading_precoded_sigma_x / (noise_power_watt + sum_power_fading_precoded_other_users_sigma_int)
        )

    info_rate_users = zeros(user_nr)

    for user_idx in range(user_nr):
        info_rate_users[user_idx] = log2(1 + sinr_users[user_idx])

    sum_rate = sum(info_rate_users)

    return sum_rate
