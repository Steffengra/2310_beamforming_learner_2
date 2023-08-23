
import numpy as np
import src


def los_channel_error_model_multiplicative_on_cos(
        error_model_config: 'src.config.config_error_model.ConfigErrorModel',
        satellite: 'src.data.satellite.Satellite',
        users: list,
        
) -> np.ndarray:

    """
    TODO: erklären - ref?
    NOTE: With this error model, satellites with ODD number of antennas will always
        have zero error on the middle antenna. Learning algorithms can exploit this to
        ignore the error.
    """

    # calculate indices for steering vectors
    steering_idx = np.arange(0, satellite.antenna_nr) - (satellite.antenna_nr - 1) / 2

 #   steering_error = np.exp(
 #       steering_idx * (
 #           1j * 2 * np.pi / satellite.wavelength
 #           * satellite.antenna_distance
 #           * satellite.rng.uniform(low=error_model_config.uniform_error_interval['low'],
 #                                   high=error_model_config.uniform_error_interval['high'],
 #                                   size=(len(users), 1))
 #       )
 #   )


    erroneous_channel_state_to_users = satellite.channel_state_to_users * satellite.steering_error

    return erroneous_channel_state_to_users
