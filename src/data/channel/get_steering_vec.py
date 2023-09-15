
import numpy as np

import src


def get_steering_vec(
        satellite: 'src.data.satellite.Satellite',
        phase_aod_steering: float,
) -> np.ndarray:
    """TODO: Comment"""

    steering_vector_to_user = np.zeros(satellite.antenna_nr, dtype='complex128')

    steering_idx = np.arange(0, satellite.antenna_nr) - (satellite.antenna_nr - 1) / 2  # todo
    # steering_idx = np.arange(0, satellite.antenna_nr)

    steering_vector_to_user[:] = np.exp(
        steering_idx
        * -1j * 2 * np.pi / satellite.wavelength
        * satellite.antenna_distance
        * phase_aod_steering
    )

    return steering_vector_to_user
