
import numpy as np

import src
from src.analysis.helpers.test_precoder_user_distance_sweep import (
    test_precoder_user_distance_sweep,
)
from src.data.precoder.mrc_precoder import (
    mrc_precoder_normalized,
)
from src.data.calc_sum_rate_no_iui import (
    calc_sum_rate_no_iui,
)


def test_mrc_precoder_user_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range: np.ndarray,
) -> None:
    """Test the MRC precoder over a range of distances with zero error."""

    def get_precoder_mrc(
        config: 'src.config.config.Config',
        satellite_manager: src.data.satellite_manager.SatelliteManager,
    ):
        w_mrc = mrc_precoder_normalized(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            **config.mrc_args,
        )

        return w_mrc

    test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='mrc',
        get_precoder_func=get_precoder_mrc,
        calc_sum_rate_func=calc_sum_rate_no_iui,
    )
