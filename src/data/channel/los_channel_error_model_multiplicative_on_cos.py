
from numpy import (
    ndarray,
    arange,
    zeros,
    exp,
    pi,
)


def los_channel_error_model_multiplicative_on_cos(
        error_model_config,
        satellite,
        users: list,
) -> ndarray:
    """
    TODO: erklären
    """

    erroneous_channel_state_to_users: ndarray = zeros((len(users), satellite.antenna_nr), dtype='complex')

    # calculate indices for steering vectors
    steering_idx = arange(0, satellite.antenna_nr) - (satellite.antenna_nr - 1) / 2

    for user in users:
        for antenna_idx in range(satellite.antenna_nr):
            steering_error = exp(
                steering_idx * (
                    1j * 2 * pi / satellite.wavelength
                    * satellite.antenna_distance
                    * satellite.rng.uniform(low=error_model_config.uniform_error_interval['low'],
                                            high=error_model_config.uniform_error_interval['high'])
                )

            )

            erroneous_channel_state_to_users[user.idx] = satellite.channel_state_to_users[user.idx] * steering_error

    return erroneous_channel_state_to_users
