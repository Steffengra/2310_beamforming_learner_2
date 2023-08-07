
import numpy as np
from keras.models import (
    load_model,
)
from pathlib import (
    Path,
)

from src.config.config import (
    Config,
)
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
        config,
        model_path,
        csit_error_sweep_range,
        monte_carlo_iterations,
) -> None:

    def get_precoder_function_learned(
        config,
        satellite_manager,
    ):
        state = config.config_learner.get_state(satellite_manager=satellite_manager, **config.config_learner.get_state_args)
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
    test_precoder_error_sweep(
        config=config,
        csit_error_sweep_range=csit_error_sweep_range,
        precoder_name='learned',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=get_precoder_function_learned,
        calc_sum_rate_func=calc_sum_rate,
    )


if __name__ == '__main__':

    cfg = Config()
    cfg.config_learner.training_name = f'sat_{cfg.sat_nr}_ant_{cfg.sat_tot_ant_nr}_usr_{cfg.user_nr}_satdist_{cfg.sat_dist_average}_usrdist_{cfg.user_dist_average}'

    iterations: int = 10_000
    sweep_range = np.arange(0.0, 0.6, 0.1)
    # sweep_range = np.arange(0.0, 1/10_000_000, 1/100_000_000)
    # sweep_range = np.arange(0, 0.07, 0.005)

    model_path = Path(
        cfg.trained_models_path,
        'sat_2_ant_4_usr_3_satdist_10000_usrdist_1000',
        'err_mult_on_steering_cos',
        'single_error',
        'error_0.1_userwiggle_30_snap_3.422',
        'model',
    )

    test_sac_precoder_error_sweep(
        config=cfg,
        model_path=model_path,
        csit_error_sweep_range=sweep_range,
        monte_carlo_iterations=iterations,
    )
