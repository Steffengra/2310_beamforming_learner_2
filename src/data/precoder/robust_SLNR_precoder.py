
import numpy as np
import scipy


def robust_SLNR_precoder_no_norm(
        channel_matrix: np.ndarray,
        autocorrelation_matrix: np.ndarray,
        noise_power_watt: float,
        power_constraint_watt: float,
) -> np.ndarray:
    """TODO: Comment"""

    user_nr = channel_matrix.shape[0]
    sat_tot_ant_nr = channel_matrix.shape[1]
    precoding_matrix = np.empty((sat_tot_ant_nr, user_nr), dtype='complex128')

    for user_idx in range(user_nr):

        weighted_autocorrelation_matrices_other_users = [
            np.matmul(  # == trace
                channel_matrix[other_user_idx, :].conj().T,
                channel_matrix[other_user_idx, :]
            )
            / sat_tot_ant_nr
            * autocorrelation_matrix[other_user_idx, :, :]
            for other_user_idx in range(user_nr) if other_user_idx != user_idx
        ]

        sum_weighted_autocorrelation_matrices_other_users = sum(weighted_autocorrelation_matrices_other_users)

        autocorrelation_matrix_user_idx = autocorrelation_matrix[user_idx, :, :]

        eigenvalues, eigenvecs = scipy.linalg.eig(
            a=autocorrelation_matrix_user_idx,
            b=(sum_weighted_autocorrelation_matrices_other_users
               + user_nr * noise_power_watt / power_constraint_watt * np.eye(sat_tot_ant_nr)),
        )
        max_eigenvalue_idx = eigenvalues.argmax()
        max_eigenvec = eigenvecs[:, max_eigenvalue_idx]

        precoding_matrix[:, user_idx] = np.sqrt(power_constraint_watt/user_nr) * max_eigenvec

    return precoding_matrix
