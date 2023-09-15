
from pathlib import Path
import gzip
import pickle

import numpy as np
from keras.models import load_model

import src
from src.analysis.helpers.test_precoder_user_distance_sweep import (
    test_precoder_user_distance_sweep,
)
from src.utils.real_complex_vector_reshaping import (
    real_vector_to_half_complex_vector,
)
from src.utils.norm_precoder import (
    norm_precoder,
)
from src.data.calc_sum_rate import (
    calc_sum_rate,
)


def test_sac_precoder_user_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range: np.ndarray,
    model_path: Path,
) -> None:
    """Test a precoder over a range of distances with zero error."""

    def get_precoder_function_learned(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
    ):

        state = config.config_learner.get_state(
            satellite_manager=satellite_manager,
            norm_factors=norm_factors,
            **config.config_learner.get_state_args
        )
        w_precoder, _ = precoder_network.call(state.astype('float32')[np.newaxis])
        w_precoder = w_precoder.numpy().flatten()

        # reshape to fit reward calculation
        w_precoder = real_vector_to_half_complex_vector(w_precoder)
        w_precoder = w_precoder.reshape((config.sat_nr * config.sat_ant_nr, config.user_nr))

        return norm_precoder(
            precoding_matrix=w_precoder,
            power_constraint_watt=config.power_constraint_watt,
            per_satellite=True,
            sat_nr=config.sat_nr,
            sat_ant_nr=config.sat_ant_nr)

    precoder_network = load_model(model_path)

    with gzip.open(Path(model_path, '..', 'config', 'norm_dict.gzip')) as file:
        norm_dict = pickle.load(file)
    norm_factors = norm_dict['norm_factors']
    if norm_factors != {}:
        config.config_learner.get_state_args['norm_state'] = True

    test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='learned',
        get_precoder_func=get_precoder_function_learned,
        calc_sum_rate_func=calc_sum_rate,
    )
