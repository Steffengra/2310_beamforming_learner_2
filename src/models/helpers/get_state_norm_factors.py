
import numpy as np

import src
from src.utils.update_sim import update_sim


def get_state_norm_factors(
    config: 'src.config.config.Config',
    satellite_manager: 'src.data.satellite_manager.SatelliteManager',
    user_manager: 'src.data.user_manager.UserManager',
) -> dict[str: str, str: dict, str: dict]:

    """
    Determines normalization factors for all streams of a
    given get_state method heuristically by sampling according to config.
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

    norm_dict['norm_factors']['means'] = np.mean(states, axis=0)
    norm_dict['norm_factors']['stds'] = np.std(states, axis=0)

    return norm_dict
