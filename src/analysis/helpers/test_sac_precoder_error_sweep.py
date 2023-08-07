
import numpy as np
from keras.models import (
    load_model,
)
from pathlib import (
    Path,
)

import src
from src.analysis.helpers.test_precoder_error_sweep import (
    test_precoder_error_sweep,
)
from src.data.calc_sum_rate import (
    calc_sum_rate,
)
from src.utils.real_complex_vector_reshaping import (
    real_vector_to_half_complex_vector,
)
from src.utils.norm_precoder import (
    norm_precoder,
)


def test_sac_precoder_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        csit_error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:

    def get_precoder_function_learned(
        cfg: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
    ):

        state = cfg.config_learner.get_state(satellite_manager=satellite_manager, **cfg.config_learner.get_state_args)
        w_precoder, _ = precoder_network.call(state.astype('float32')[np.newaxis])
        w_precoder = w_precoder.numpy().flatten()

        # reshape to fit reward calculation
        w_precoder = real_vector_to_half_complex_vector(w_precoder)
        w_precoder = w_precoder.reshape((cfg.sat_nr * cfg.sat_ant_nr, cfg.user_nr))

        return norm_precoder(
            precoding_matrix=w_precoder,
            power_constraint_watt=cfg.power_constraint_watt,
            per_satellite=True,
            sat_nr=cfg.sat_nr,
            sat_ant_nr=cfg.sat_ant_nr)

    precoder_network = load_model(model_path)
    test_precoder_error_sweep(
        config=config,
        csit_error_sweep_range=csit_error_sweep_range,
        precoder_name='learned',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=get_precoder_function_learned,
        calc_sum_rate_func=calc_sum_rate,
    )
