
import numpy as np
import src

from src.data.channel.get_steering_vec import get_steering_vec


def los_channel_model(
        satellite: 'src.data.satellite.Satellite',
        users: list,
        error_free: bool= False
) -> np.ndarray:

    """
    The los channel model calculates complex csi for one satellite to all users from
        1) amplitude dampening based on sat gain, user gain, and freq-dependent distance gain
        2) phase shift by freq-dependent distance
        3) phase shift by satellite steering vectors
    TODO: describe this - correct? reference?
    """

    if error_free is True:

        errors = {
            'additive_error_on_overall_phase_shift': np.zeros(len(users)),
            'additive_error_on_aod': np.zeros(len(users)),
            'additive_error_on_cosine_of_aod': np.zeros(len(users)),
            'additive_error_on_channel_vector': np.zeros(len(users)),
        }
        
    else:
        
        errors = satellite.estimation_errors

    channel_state_information = np.zeros((len(users), satellite.antenna_nr), dtype='complex128')

    for user_idx,user in enumerate(users):
        power_ratio = (
                satellite.antenna_gain_linear
                * user.gain_linear
                * (satellite.wavelength / (4 * np.pi * satellite.distance_to_users[user.idx])) ** 2
        )
        amplitude_damping = np.sqrt(power_ratio)

        phase_shift = satellite.distance_to_users[user.idx] % satellite.wavelength * 2 * np.pi / satellite.wavelength
        phase_shift_error = errors['additive_error_on_overall_phase_shift'][user_idx]
        phase_aod_steering = np.cos(satellite.aods_to_users[user_idx] + errors['additive_error_on_aod'][user_idx]) + errors['additive_error_on_cosine_of_aod'][user_idx]

        steering_vector_to_user = get_steering_vec(
            satellite,
            phase_aod_steering
        )

        channel_state_information[user.idx, :] = (
            amplitude_damping
            * np.exp(-1j * (phase_shift + phase_shift_error))
            * steering_vector_to_user
        )

    return channel_state_information
