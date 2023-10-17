
import numpy as np

import src
from src.utils.update_sim import update_sim
from src.models.helpers.get_state import (
    get_state_erroneous_channel_state_information,
    get_state_aods,
)


def get_state_norm_factors(
    config: 'src.config.config.Config',
    satellite_manager: 'src.data.satellite_manager.SatelliteManager',
    user_manager: 'src.data.user_manager.UserManager',
) -> dict[str: str, str: dict, str: dict]:

    """
    Determines normalization factors for a given get_state method heuristically by sampling
        according to config.
    """

    # define default norm_dict
    norm_dict: dict = {
        'get_state_method': str(config.config_learner.get_state),
        'get_state_args': config.config_learner.get_state_args,
        'norm_factors': {},
    }

    # if no norm, don't determine norm factors
    if not config.config_learner.get_state_args['norm_state']:
        return norm_dict

    # set get_state norm argument to false for the sampling process
    get_state_args = config.config_learner.get_state_args.copy()
    get_state_args['norm_state'] = False

    update_sim(config=config, satellite_manager=satellite_manager, user_manager=user_manager)

    # gather state samples
    states = []
    for _ in range(config.config_learner.get_state_norm_factors_iterations):

        state = config.config_learner.get_state(satellite_manager=satellite_manager, **get_state_args)
        states.append(state)
        update_sim(config=config, satellite_manager=satellite_manager, user_manager=user_manager)

    # determine norm factors according to get_state method
    if config.config_learner.get_state == get_state_erroneous_channel_state_information:

        if get_state_args['csi_format'] == 'rad_phase':

            states_radius = np.array([state[:int(len(state)/2)] for state in states]).flatten()
            states_phase = np.array([state[int(len(state)/2):] for state in states]).flatten()

            norm_dict['norm_factors']['radius_mean'] = np.mean(states_radius)
            norm_dict['norm_factors']['radius_std'] = np.std(states_radius)
            norm_dict['norm_factors']['phase_mean'] = np.mean(states_phase)
            norm_dict['norm_factors']['phase_std'] = np.std(states_phase)
            # note: statistical analysis has shown that the means, especially of phase,
            #  take a lot of iterations to determine with confidence. Hence, we might only use std for norm.

        elif get_state_args['csi_format'] == 'rad_phase_reduced':
            num_users = satellite_manager.satellites[0].user_nr
            num_satellites = len(satellite_manager.satellites)
            states_radius = np.array([state[:num_users * num_satellites] for state in states]).flatten()
            states_phase = np.array([state[num_users * num_satellites:] for state in states]).flatten()

            norm_dict['norm_factors']['radius_mean'] = np.mean(states_radius)
            norm_dict['norm_factors']['radius_std'] = np.std(states_radius)
            norm_dict['norm_factors']['phase_mean'] = np.mean(states_phase)
            norm_dict['norm_factors']['phase_std'] = np.std(states_phase)

        elif get_state_args['csi_format'] == 'real_imag':

            states_real_imag = np.array(states).flatten()

            norm_dict['norm_factors']['mean'] = np.mean(states_real_imag)
            norm_dict['norm_factors']['std'] = np.std(states_real_imag)

        else:

            raise ValueError('unknown csi_format')

    elif config.config_learner.get_state == get_state_aods:

        states_aods = np.array(states).flatten()

        norm_dict['norm_factors']['mean'] = np.mean(states_aods)
        norm_dict['norm_factors']['std'] = np.std(states_aods)

    else:

        raise ValueError('unknown get_state function')

    return norm_dict
