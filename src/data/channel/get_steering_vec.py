import numpy as np
import src

def get_steering_vec(
        satellite,
        phase_aod_steering
):

    steering_vector_to_user = np.zeros((satellite.antenna_nr), dtype='complex128')

    #steering_idx = np.arange(0, satellite.antenna_nr) - (satellite.antenna_nr - 1) / 2
    steering_idx = np.arange(0,satellite.antenna_nr)
    print(steering_idx)
    print(steering_vector_to_user.shape)
    print(phase_aod_steering)

    steering_vector_to_user[:] = np.exp(
            steering_idx * (
            -1j * 2 * np.pi / satellite.wavelength
            * satellite.antenna_distance
            * phase_aod_steering)
        )

    return steering_vector_to_user

    #satellite.steering_vectors_to_users = np.zeros((len(users), satellite.antenna_nr), dtype='complex128')

    #steering_idx = np.arange(0, satellite.antenna_nr) - (satellite.antenna_nr - 1) / 2
    #steering_idx = np.arange(0,satellite.antenna_nr)
    #print(steering_idx)

    #for user in users:
    #    satellite.steering_vectors_to_users[user.idx, :] = np.exp(
    #        steering_idx * 
    #        -1j * 2 * np.pi / satellite.wavelength
    #        * satellite.antenna_distance
    #        * phase_aod_steering
    #    )
        