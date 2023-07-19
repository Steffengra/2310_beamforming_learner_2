
from numpy import (
    arange,
)

from src.config.config import (
    Config,
)
from src.analysis.helpers.test_precoder_error_sweep import (
    test_precoder_error_sweep,
)
from src.data.precoder.mrc_precoder import (
    mrc_precoder_normalized,
)
from src.data.calc_sum_rate_no_iui import (
    calc_sum_rate_no_iui,
)


def test_mrc_precoder_error_sweep(
        config,
        csit_error_sweep_range,
        monte_carlo_iterations,
) -> None:

    def get_precoder_mrc(
        config,
        satellite_manager,
    ):
        w_mrc = mrc_precoder_normalized(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            **config.mrc_args,
        )

        return w_mrc

    test_precoder_error_sweep(
        config=config,
        csit_error_sweep_range=csit_error_sweep_range,
        precoder_name='mrc',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=get_precoder_mrc,
        calc_sum_rate_func=calc_sum_rate_no_iui,
    )


if __name__ == '__main__':

    cfg = Config()
    cfg.config_learner.training_name = f'sat_{cfg.sat_nr}_ant_{cfg.sat_tot_ant_nr}_usr_{cfg.user_nr}_satdist_{cfg.sat_dist_average}_usrdist_{cfg.user_dist_average}'

    iterations: int = 10_000
    sweep_range = arange(0.0, 0.6, 0.1)
    # sweep_range = arange(0.0, 1/10_000_000, 1/100_000_000)

    test_mrc_precoder_error_sweep(
        config=cfg,
        csit_error_sweep_range=sweep_range,
        monte_carlo_iterations=iterations,
    )
