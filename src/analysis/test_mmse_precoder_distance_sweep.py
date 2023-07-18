
from numpy import (
    arange,
)

from src.config.config import (
    Config,
)
from src.analysis.helpers.test_precoder_distance_sweep import (
    test_precoder_distance_sweep,
)
from src.data.precoder.mmse_precoder import (
    mmse_precoder_normalized,
)
from src.data.calc_sum_rate import (
    calc_sum_rate,
)


def test_mmse_precoder_distance_sweep(
    config,
    distance_sweep_range,
) -> None:

    def get_precoder_mmse(
        config,
        satellite_manager,
    ):

        w_mmse = mmse_precoder_normalized(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            **config.mmse_args,
        )

        return w_mmse

    test_precoder_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='mmse',
        get_precoder_func=get_precoder_mmse,
        calc_sum_rate_func=calc_sum_rate,
    )


if __name__ == '__main__':

    cfg = Config()
    cfg.config_learner.training_name = f'sat_{cfg.sat_nr}_ant_{cfg.sat_tot_ant_nr}_usr_{cfg.user_nr}_satdist_{cfg.sat_dist_average}_usrdist_{cfg.user_dist_average}'

    sweep_range = arange(50_000-30, 50_000+30, 0.01)

    test_mmse_precoder_distance_sweep(config=cfg, distance_sweep_range=sweep_range)
