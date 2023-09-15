
import numpy as np


def norm_precoder(
        precoding_matrix: np.ndarray,
        power_constraint_watt: float or int,
        per_satellite: bool,
        sat_nr: int = 1,
        sat_ant_nr: int = 1,
) -> np.ndarray:
    """
    Normalizes precoding matrix of dimension (sat_nr * ant_nr, user_nr)
    with sat 1 ant 1, sat1 ant 2, sat1 ant 3, sat 2 ant 1...

    tr(A^H * A) is the sum of squared elements
    after applying norm_factor, the trace of norm_factor * (A^H * A) will be == power_constraint_watt
    """

    normalized_precoder = np.empty(shape=precoding_matrix.shape, dtype='complex128')

    if per_satellite:

        # normalize to (power_constraint_watt / sat_nr) for each satellite
        for satellite_id in range(sat_nr):

            satellite_index_start = satellite_id * sat_ant_nr
            w_precoder_slice = precoding_matrix[satellite_index_start:satellite_index_start + sat_ant_nr, :]

            norm_factor_slice = np.sqrt(
                power_constraint_watt / sat_nr / np.trace(np.matmul(w_precoder_slice.conj().T, w_precoder_slice))
            )
            w_precoder_slice_normed = norm_factor_slice * w_precoder_slice

            normalized_precoder[satellite_index_start:satellite_index_start + sat_ant_nr, :] = w_precoder_slice_normed.copy()  # todo: is copy required here?

    else:

        # normalize to power constraint
        norm_factor = np.sqrt(power_constraint_watt / np.trace(np.matmul(precoding_matrix.conj().T, precoding_matrix)))
        normalized_precoder = norm_factor * precoding_matrix

    return normalized_precoder
