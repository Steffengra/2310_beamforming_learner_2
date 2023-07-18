
from numpy import (
    ndarray,
    sqrt,
    matmul,
    trace,
)


def norm_precoder(
        precoding_matrix: ndarray,
        power_constraint_watt: float or int,
        per_satellite: bool,
        sat_nr: int = 1,
        sat_ant_nr: int = 1,
) -> ndarray:
    """
    normalizes precoding matrix of dimension (sat_nr * ant_nr, user_nr)
    with sat 1 ant 1, sat1 ant 2, sat1 ant 3, sat 2 ant 1...

    tr(A^H * A) is the sum of squared elements
    after applying norm_factor, the trace of norm_factor * (A^H * A) will be == power_constraint_watt
    """

    if per_satellite:

        # normalize to (power_constraint_watt / sat_nr) for each satellite
        for satellite_id in range(sat_nr):

            satellite_index_start = satellite_id * sat_ant_nr
            w_precoder_slice = precoding_matrix[satellite_index_start:satellite_index_start + sat_ant_nr, :]

            norm_factor_slice = sqrt(
                power_constraint_watt / sat_nr / trace(matmul(w_precoder_slice.conj().T, w_precoder_slice))
            )
            w_precoder_slice_normed = norm_factor_slice * w_precoder_slice

            precoding_matrix[satellite_index_start:satellite_index_start + sat_ant_nr, :] = w_precoder_slice_normed

        normalized_precoder = precoding_matrix

    else:

        # normalize to power constraint
        norm_factor = sqrt(power_constraint_watt / trace(matmul(precoding_matrix.conj().T, precoding_matrix)))
        normalized_precoder = norm_factor * precoding_matrix

    return normalized_precoder
