
from numpy import (
    ndarray,
    finfo,
    eye,
    matmul,
    trace,
    sqrt,
)
from numpy.linalg import (
    inv,
)
import numpy as np


def mmse_precoder(
        channel_matrix,
        noise_power_watt: float,
        power_constraint_watt: float,
) -> ndarray:

    # inversion_constant_lambda = finfo('float32').tiny
    inversion_constant_lambda = 0

    user_nr = channel_matrix.shape[0]
    sat_tot_ant_nr = channel_matrix.shape[1]

    precoding_matrix = (
        matmul(
            inv(
                matmul(channel_matrix.conj().T, channel_matrix)
                + (noise_power_watt * user_nr / power_constraint_watt + inversion_constant_lambda) * eye(sat_tot_ant_nr)
            ),
            channel_matrix.conj().T
        )
    )

    norm_factor = sqrt(power_constraint_watt / trace(matmul(precoding_matrix.conj().T, precoding_matrix)))

    w_mmse = norm_factor * precoding_matrix

    return w_mmse
