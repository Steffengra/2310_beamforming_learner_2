
import numpy as np


def mrc_precoder_normalized(
    channel_matrix: np.ndarray,
    power_constraint_watt: float,
) -> np.ndarray:
    """TODO: Comment"""

    user_nr = channel_matrix.shape[0]
    sat_tot_ant_nr = channel_matrix.shape[1]

    w_mrc = np.empty((sat_tot_ant_nr, user_nr), dtype='complex128')

    for user_id in range(user_nr):

        H_k = channel_matrix[user_id, :]

        w = (1 / np.linalg.norm(H_k)) * H_k.conj().T * np.sqrt(power_constraint_watt)

        w_mrc[:, user_id] = w

    return w_mrc
