
import numpy as np

import src
from src.utils.real_complex_vector_reshaping import (
    complex_vector_to_double_real_vector,
    complex_vector_to_rad_and_phase,
)


def get_state_erroneous_channel_state_information(
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        csi_format: str,
        norm_state: bool,
        norm_factors: dict = None,
) -> np.ndarray:

    # FOR CONFIG:
    # self.get_state_args = {
    #     'csi_format': 'rad_phase',  # 'rad_phase', 'real_imag'
    #     'norm_state': True,  # !!HEURISTIC!!, this will break if you dramatically change the setup
    # }

    if norm_state and norm_factors is None:
        raise ValueError('no norm factors provided')

    erroneous_csi = satellite_manager.erroneous_channel_state_information.flatten()

    if csi_format == 'rad_phase':
        # TODO: Heuristic standardization for this falls apart when inter user distances or inter satellite
        #  distances are changed significantly
        state_real = complex_vector_to_rad_and_phase(erroneous_csi)
        if norm_state:

            half_length_idx = int(len(state_real) / 2)

            # normalize radius
            # heuristic standardization
            state_real[:half_length_idx] -= norm_factors['means'][:half_length_idx]  # needs a moderate amount of samples
            state_real[:half_length_idx] /= norm_factors['stds'][:half_length_idx]  # needs few samples

            # normalize phase
            # heuristic standardization
            # state_real[half_length_idx:] -= norm_factors['means'][half_length_idx:]  # needs A LOT of samples
            state_real[half_length_idx:] /= norm_factors['stds'][half_length_idx:]  # needs few samples

    elif csi_format == 'real_imag':
        state_real = complex_vector_to_double_real_vector(erroneous_csi)
        if norm_state:

            # state_real *= 1e7  # roughly range [-1, 1]

            # VERY heuristic standardization
            # state_real -= -4.308892163699242e-09
            # state_real /= 7.015404816259004e-08

            # heuristic standardization
            # state_real -= norm_factors['means']  # needs a lot of samples, todo: should be 0?
            state_real /= norm_factors['stds']  # needs few samples

    else:
        raise ValueError(f'Unknown CSI Format {csi_format}')

    return state_real


def get_state_aods(
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_state: bool,
        norm_factors: dict = None,
) -> np.ndarray:

    # TODO: Heuristic standardization for this falls apart when inter user distances or inter satellite
    #  distances are changed significantly

    # FOR CONFIG:
    # self.get_state_args = {
    #     'norm_state': True,  # !!HEURISTIC!!, this will break if you dramatically change the setup
    # }

    if norm_state and norm_factors is None:
        raise ValueError('no norm factors provided')

    state = satellite_manager.get_aods_to_users().flatten()
    if norm_state:

        # heuristic standardization
        state -= norm_factors['means']
        state /= norm_factors['stds']

    return state.flatten()
