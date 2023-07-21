from numpy import (
    ndarray,
    array,
    arange,
    newaxis,
    zeros,
    sqrt,
    exp,
    pi,
)

from src.data.satellite import (
    Satellite,
)


def los_channel_error_model_in_sat_and_user_pos(
        error_model_config,
        satellite: Satellite,
        users: list,
) -> ndarray:
    """
    TODO: describe this - ref?
    """

    channel_state_information = zeros((len(users), satellite.antenna_nr), dtype='complex128')
    for user in users:

        power_ratio = (
                satellite.antenna_gain_linear
                * user.gain_linear
                * (satellite.wavelength / (4 * pi * satellite.distance_to_users[user.idx])) ** 2
        )
        amplitude_damping = sqrt(power_ratio)

        phase_shift = satellite.distance_to_users[user.idx] % satellite.wavelength * 2 * pi / satellite.wavelength
        phase_shift_error = 2 * pi / satellite.wavelength * satellite.rng.normal(loc=0, scale=error_model_config.phase_sat_error_std)

        channel_state_information[user.idx, :] = (
            amplitude_damping
            * exp(1j * phase_shift) 
            * exp(1j * phase_shift_error)
            * satellite.steering_vectors_to_users[user.idx]
        )

    # calculate indices for steering vectors
    steering_idx = arange(0, satellite.antenna_nr) - (satellite.antenna_nr - 1) / 2

    steering_error = exp(
        steering_idx * (
            1j * 2 * pi / satellite.wavelength
            * satellite.antenna_distance
            * satellite.rng.uniform(low=error_model_config.uniform_error_interval['low'],
                                    high=error_model_config.uniform_error_interval['high'],
                                    size=(len(users), 1))
        )
    )
    erroneous_channel_state_to_users = channel_state_information * steering_error

    return erroneous_channel_state_to_users
