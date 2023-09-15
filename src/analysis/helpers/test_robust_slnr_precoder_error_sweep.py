
import numpy as np

import src
from src.analysis.helpers.test_precoder_error_sweep import (
    test_precoder_error_sweep,
)
from src.data.precoder.robust_SLNR_precoder import (
    robust_SLNR_precoder_no_norm,
)
from src.data.precoder.calc_autocorrelation import (
    calc_autocorrelation,
)
from src.data.calc_sum_rate import (
    calc_sum_rate,
)


def test_robust_slnr_precoder_error_sweep(
        config: 'src.config.config.Config',
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:
    """Test the robust SLNR precoder for a range of error configuration with monte carlo average."""

    def get_precoder_robust_slnr(
            config: 'src.config.config.Config',
            satellite_manager: 'src.data.satellite_manager.SatelliteManager',
    ) -> np.ndarray:
        autocorrelation = calc_autocorrelation(
            satellite=satellite_manager.satellites[0],
            error_model_config=config.config_error_model,
            error_distribution='uniform',
        )

        w_robust_slnr = robust_SLNR_precoder_no_norm(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            autocorrelation_matrix=autocorrelation,
            noise_power_watt=config.noise_power_watt,
            power_constraint_watt=config.power_constraint_watt,
        )

        return w_robust_slnr

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='robust_slnr',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=get_precoder_robust_slnr,
        calc_sum_rate_func=calc_sum_rate,
    )
