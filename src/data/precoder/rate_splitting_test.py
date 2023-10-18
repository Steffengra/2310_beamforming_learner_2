
import numpy as np
from numpy.linalg import (
    inv,
)

from src.utils.norm_precoder import (
    norm_precoder,
)
from src.data.precoder.mmse_precoder import (
    mmse_precoder_normalized,
)


# WIP


def rate_splitting_test_no_norm(
        channel_matrix,
        noise_power_watt: float,
        power_constraint_watt: float,
        rsma_factor: float,
) -> np.ndarray:

    # inversion_constant_lambda = finfo('float32').tiny
    inversion_constant_lambda = 0

    user_nr = channel_matrix.shape[0]
    sat_tot_ant_nr = channel_matrix.shape[1]

    w_common = np.ones()

    w_private = mmse_precoder_normalized(
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        **config.mmse_args
    )

    return precoding_matrix
