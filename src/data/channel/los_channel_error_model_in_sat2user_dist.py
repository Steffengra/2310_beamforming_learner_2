
from numpy import (
    ndarray,
    zeros,
    sqrt,
    exp,
    pi,
)

from src.data.satellite import (
    Satellite,
)


def los_channel_error_model_in_sat2user_dist(
        error_model_config,
        satellite: Satellite,
        users: list,
) -> ndarray:
    """
    This error model calculates an erroneous channel state information estimate based on a
    perturbed sat2user distance estimate.
    """

    erroneous_channel_state_to_users = zeros((len(users), satellite.antenna_nr), dtype='complex')

    for user in users:

        # Perturb the satellite-to-user distance according to config
        satellite_to_user_distance_estimate = (
                satellite.distance_to_users[user.idx]
                * (1 + satellite.rng.normal(loc=0, scale=error_model_config.distance_error_variance))
        )

        # Calculate channel state estimation based on the perturbed distance estimate
        power_ratio = (
                satellite.antenna_gain_linear
                * user.gain_linear
                * (satellite.wavelength / (4 * pi * satellite_to_user_distance_estimate)) ** 2
        )
        amplitude_damping = sqrt(power_ratio)

        phase_shift = satellite_to_user_distance_estimate % satellite.wavelength * 2 * pi / satellite.wavelength

        erroneous_channel_state_to_users[user.idx, :] = (
            amplitude_damping
            * exp(1j * phase_shift)
            * satellite.steering_vectors_to_users[user.idx]
        )

    return erroneous_channel_state_to_users
