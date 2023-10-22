
import numpy as np

import src
from src.utils.get_wavelength import (
    get_wavelength,
)


def calc_autocorrelation(
        satellite,
        error_model_config: 'src.config.config_error_model.ConfigErrorModel',
        error_distribution: str,
) -> np.ndarray:
    """TODO: Comment"""

    wavelength = get_wavelength(satellite.freq)
    wavenumber = 2 * np.pi / wavelength
    user_nr = satellite.user_nr
    sat_ant_nr = satellite.antenna_nr
    sat_antenna_spacing = satellite.antenna_distance
    errors = satellite.estimation_errors

    autocorrelation_matrix = np.zeros((user_nr, sat_ant_nr, sat_ant_nr), dtype='complex128')

    for user_id in range(user_nr):

        phi_dach = (
                np.cos(
                    satellite.aods_to_users[user_id]
                    + errors['additive_error_on_aod'][user_id]
                )
                + errors['additive_error_on_cosine_of_aod'][user_id]
        )

        # calc characteristic function all at once for performance reasons
        antenna_shift_idxs = np.empty((sat_ant_nr, sat_ant_nr))
        for antenna_row_id in range(sat_ant_nr):
            for antenna_col_id in range(sat_ant_nr):

                antenna_shift_idxs[antenna_row_id, antenna_col_id] = (
                    wavenumber * sat_antenna_spacing
                    * (antenna_col_id - antenna_row_id)
                )

        if error_distribution == 'uniform':
            characteristic_functions = np.sinc(
                antenna_shift_idxs
                * error_model_config.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high']
                / np.pi
            )

        elif error_distribution == 'gaussian':
            characteristic_functions = np.exp(
                -antenna_shift_idxs**2
                * error_model_config.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['scale']**2
                / 2
            )

        else:
            raise ValueError('Unknown error distribution on cosine of AODs')

        # calc autocorrelation matrix entries
        for antenna_row_id in range(sat_ant_nr):

            for antenna_col_id in range(sat_ant_nr):
                autocorrelation_matrix[user_id, antenna_row_id, antenna_col_id] = (
                        np.exp(1j * wavenumber * sat_antenna_spacing * (antenna_row_id - antenna_col_id) * phi_dach)
                        * characteristic_functions[antenna_row_id, antenna_col_id]
                )

    return autocorrelation_matrix
