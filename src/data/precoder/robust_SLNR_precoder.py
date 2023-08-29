
import numpy as np
import scipy


def robust_SLNR_precoder_no_norm(
        channel_matrix: np.ndarray,
        autocorrelation_matrix: np.ndarray,
        noise_power_watt: float,
        power_constraint_watt: float,
) -> np.ndarray:

    user_nr = channel_matrix.shape[0]
    sat_tot_ant_nr = channel_matrix.shape[1]
    precoding_matrix = np.empty((sat_tot_ant_nr, user_nr), dtype='complex128')

    # eigenvalues_channel, eingenvecs_channel = np.linalg.eig(np.matmul(channel_matrix, channel_matrix.conj().T))
    # print('Eigenvalues channel')
    # print(eigenvalues_channel)
    # print('Eigenvalues ende')

    for user_idx in range(user_nr):

        channel_vec_user_idx = channel_matrix[user_idx, :]

        power_channel_user_idx = (
                np.matmul(channel_vec_user_idx.conj(), channel_vec_user_idx)
                / sat_tot_ant_nr
        )
        # == trace

        weighted_autocorrelation_matrices_other_users = [
            np.matmul(  # == trace
                channel_matrix[other_user_idx, :].conj().T,
                channel_matrix[other_user_idx, :]
            )  # TODO: ????????????????
            / sat_tot_ant_nr
            * autocorrelation_matrix[other_user_idx, :, :]
            for other_user_idx in range(user_nr) if other_user_idx != user_idx
        ]

        sum_weighted_autocorrelation_matrices_other_users = sum(weighted_autocorrelation_matrices_other_users)

        autocorrelation_matrix_user_idx = autocorrelation_matrix[user_idx, :, :]

        generalized_Rayleigh_quotient = (
            np.linalg.inv(
                sum_weighted_autocorrelation_matrices_other_users
                + user_nr * noise_power_watt / power_constraint_watt * np.eye(sat_tot_ant_nr)
            )
            * power_channel_user_idx  # TODO: ?????
            * autocorrelation_matrix_user_idx
        )

        # with np.printoptions(linewidth=150):
        #     print(generalized_Rayleigh_quotient)
        #     print(scipy.linalg.ishermitian(generalized_Rayleigh_quotient, atol=1e-16))
        # exit()

        eigenvalues, eigenvecs = np.linalg.eig(generalized_Rayleigh_quotient)
        max_eigenvalue_idx = eigenvalues.argmax()
        max_eigenvec = eigenvecs[:, max_eigenvalue_idx]

        # print('.eig', eigenvalues)
        # print('norm', np.linalg.norm(max_eigenvec), '\n')
        # print(eigenvecs)

        eigenvalues2, eigenvecs2 = scipy.linalg.eig(
            a=autocorrelation_matrix_user_idx * power_channel_user_idx,
            b=(sum_weighted_autocorrelation_matrices_other_users + user_nr * noise_power_watt / power_constraint_watt * np.eye(sat_tot_ant_nr)),
        )
        max_eigenvalue2_idx = eigenvalues2.argmax()
        max_eigenvec2 = eigenvecs2[:, max_eigenvalue2_idx]
        # print('generalized', eigenvalues2)
        # print('norm', np.linalg.norm(max_eigenvec2), '\n')

        eigenvalues3, eigenvecs3 = np.linalg.eigh(generalized_Rayleigh_quotient)

        max_eigenvalue3_idx = eigenvalues3.argmax()
        max_eigenvec3 = eigenvecs3[:, max_eigenvalue3_idx]
        # print('.eigh', eigenvalues3)
        # print('norm', np.linalg.norm(max_eigenvec3), '\n')
        # print(eigenvecs3)

        precoding_matrix[:, user_idx] = np.sqrt(power_constraint_watt/user_nr) * max_eigenvec2

    return precoding_matrix
