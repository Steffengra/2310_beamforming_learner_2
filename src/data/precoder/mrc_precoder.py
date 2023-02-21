
from numpy import (
    empty,
    sqrt,
)
from numpy.linalg import (
    norm,
)


def mrc_precoder_normalized(
        channel_matrix,
        power_constraint_watt: float,
):

    user_nr = channel_matrix.shape[0]
    sat_tot_ant_nr = channel_matrix.shape[1]

    w_mrc = empty((sat_tot_ant_nr, user_nr), dtype='complex')

    for user_id in range(user_nr):

        H_k = channel_matrix[user_id, :]

        w = (1 / norm(H_k)) * H_k.conj().T * sqrt(power_constraint_watt)

        w_mrc[:, user_id] = w

    return w_mrc
