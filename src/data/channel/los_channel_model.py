
import numpy as np
import src


def los_channel_model(
        satellite: 'src.data.satellite.Satellite',
        users: list,
) -> np.ndarray:

    """
    The los channel model calculates complex csi for one satellite to all users from
        1) amplitude dampening based on sat gain, user gain, and freq-dependent distance gain
        2) phase shift by freq-dependent distance
        3) phase shift by satellite steering vectors
    TODO: describe this - correct? reference?
    """

    channel_state_information = np.zeros((len(users), satellite.antenna_nr), dtype='complex128')
    for user in users:
        power_ratio = (
                satellite.antenna_gain_linear
                * user.gain_linear
                * (satellite.wavelength / (4 * np.pi * satellite.distance_to_users[user.idx])) ** 2
        )
        amplitude_damping = np.sqrt(power_ratio)

        phase_shift = satellite.distance_to_users[user.idx] % satellite.wavelength * 2 * np.pi / satellite.wavelength

        channel_state_information[user.idx, :] = (
            amplitude_damping
            * np.exp(1j * phase_shift)
            * satellite.steering_vectors_to_users[user.idx]
        )

    return channel_state_information
