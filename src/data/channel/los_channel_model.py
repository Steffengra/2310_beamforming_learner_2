
from numpy import (
    ndarray,
    array,
    newaxis,
    zeros,
    sqrt,
    exp,
    pi,
)

from src.data.satellite import (
    Satellite,
)


def los_channel_model(
        satellite: Satellite,
        users: list,
) -> ndarray:
    """
    TODO: describe this
    """

    amplitude_dampening = [
        sqrt(
            satellite.antenna_gain_linear
            * user.gain_linear
            * (satellite.wavelength / (4 * pi * satellite.distance_to_users[user.idx])) ** 2
        )
        for user in users
    ]
    phase_shift = satellite.distance_to_users % satellite.wavelength * 2 * pi / satellite.wavelength
    channel_state_information = (
            amplitude_dampening
            * exp(1j * phase_shift)[:, None]
            * satellite.steering_vectors_to_users
    )

    return channel_state_information
