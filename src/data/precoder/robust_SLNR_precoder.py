
import numpy as np

from src.utils.norm_precoder import (
    norm_precoder,
)

# only works reliable if we have an uniform error > 1e-4

def robust_SLNR_precoder_no_norm(
        channel_matrix: np.ndarray,
        autocorrelation_matrix, 
        noise_power_watt: float,
        power_constraint_watt: float,
) -> np.ndarray:

    user_nr = channel_matrix.shape[0]
    sat_tot_ant_nr = channel_matrix.shape[1]
    precoding_matrix = np.empty((sat_tot_ant_nr, user_nr), dtype='complex128')

    eigenvalues_channel, eingenvecs_channel = np.linalg.eig(np.matmul(channel_matrix,channel_matrix.conj().T))
    print('Eigenvalues channel')
    print(eigenvalues_channel)
    print('Eigenvalues ende')

    for user_idx in range(user_nr):

        channel_vec_user_idx = channel_matrix[user_idx, :][np.newaxis]
        print(channel_vec_user_idx)

        power_channel_user_idx = np.trace(np.matmul(channel_vec_user_idx.conj().T, channel_vec_user_idx ))

        autocorrelation_matrix_user_idx = autocorrelation_matrix[user_idx,:,:]

        weighted_autocorrelation_matrices_other_users = [
            np.trace(np.matmul(channel_matrix[other_user_idx, :][np.newaxis].conj().T  , channel_matrix[other_user_idx, :][np.newaxis])) * autocorrelation_matrix[other_user_idx,:,:]
            for other_user_idx in range(user_nr) if other_user_idx != user_idx
        ]

        sum_weighted_autocorrelation_matrices_other_users = sum(weighted_autocorrelation_matrices_other_users)
        #print('autocorrelation matrix other users')
        #print(sum_weighted_autocorrelation_matrices_other_users.shape)
        generalized_Rayleigh_quotient = (np.linalg.inv(sum_weighted_autocorrelation_matrices_other_users + user_nr * noise_power_watt/power_constraint_watt * np.eye(sat_tot_ant_nr, sat_tot_ant_nr))
                                            * power_channel_user_idx * autocorrelation_matrix_user_idx)
        
        eigenvalues, eigenvecs = np.linalg.eig(generalized_Rayleigh_quotient)

        max_eigenvalue_idx = eigenvalues.argmax()

        max_eigenvec = eigenvecs[max_eigenvalue_idx]
        print(max_eigenvec)

        precoding_matrix[:,user_idx] = np.sqrt(power_constraint_watt/user_nr) * max_eigenvec

    return precoding_matrix