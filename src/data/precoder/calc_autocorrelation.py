
import numpy as np

import src
from src.utils.get_wavelength import (
    get_wavelength,
)


def calc_autocorrelation(
        steering_vec_estimate: np.ndarray,
        sat_antenna_spacing: float,
        carrier_frequency: float,
        error_model_config: 'src.config.config_error_model.ConfigErrorModel',
        error_distribution: str,
        gaussian_error_std: float or None = None
) -> np.ndarray:

    wavelength = get_wavelength(carrier_frequency)
    wavenumber = 2 * np.pi / wavelength
    user_nr = steering_vec_estimate.shape[0]
    sat_ant_nr = steering_vec_estimate.shape[1]
    autocorrelation_matrix = np.zeros((user_nr, sat_ant_nr, sat_ant_nr), dtype='complex128')

    for user_id in range(user_nr):

        phi_dach = (
                np.angle(steering_vec_estimate[user_id, 0])
                / (wavenumber * sat_antenna_spacing * (1 - sat_ant_nr) / 2)
        )

        for antenna_row_id in range(sat_ant_nr):

            for antenna_col_id in range(sat_ant_nr):

                antenna_shift_idx = (
                        wavenumber * sat_antenna_spacing
                        * (antenna_col_id - antenna_row_id)
                )

                if error_distribution == 'uniform':
                    characteristic_function = np.sinc(
                        antenna_shift_idx * error_model_config.uniform_error_interval['high'] / np.pi
                    )

                elif error_distribution == 'gaussian':
                    characteristic_function = np.exp(-antenna_shift_idx**2 * gaussian_error_std**2 / 2)

                else:
                    raise ValueError('Unknown error distribution on cosine of AODs')

                autocorrelation_matrix[user_id, antenna_row_id, antenna_col_id] = (
                        np.exp(-1j * wavenumber * sat_antenna_spacing * (antenna_row_id - antenna_col_id) * phi_dach)
                        * characteristic_function
                )

    return autocorrelation_matrix
