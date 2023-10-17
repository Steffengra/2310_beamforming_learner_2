
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
    """TODO: Comment"""

    # FOR CONFIG:
    # self.get_state_args = {
    #     'csi_format': 'rad_phase',  # 'rad_phase', 'real_imag'
    #     'norm_state': True,  # !!HEURISTIC!!, this will break if you dramatically change the setup
    # }

    if norm_state and norm_factors is None:
        raise ValueError('no norm factors provided')

    erroneous_csi = satellite_manager.erroneous_channel_state_information.flatten()

    if csi_format == 'rad_phase':

        state_real = complex_vector_to_rad_and_phase(erroneous_csi)
        if norm_state:

            half_length_idx = int(len(state_real) / 2)

            # normalize radius
            # heuristic standardization
            state_real[:half_length_idx] -= norm_factors['radius_mean']  # needs a moderate amount of samples
            state_real[:half_length_idx] /= norm_factors['radius_std']  # needs few samples

            # normalize phase
            # heuristic standardization
            # state_real[half_length_idx:] -= norm_factors['phase_mean']  # needs A LOT of samples
            state_real[half_length_idx:] /= norm_factors['phase_std']  # needs few samples

    # like rad_phase, but only one radius per satellite+user. Rationale: path loss dominates as d_satuser >> d_antenna
    elif csi_format == 'rad_phase_reduced':

        state_real = complex_vector_to_rad_and_phase(erroneous_csi)
        if norm_state:

            half_length_idx = int(len(state_real) / 2)

            # normalize radius
            # heuristic standardization
            state_real[:half_length_idx] -= norm_factors['radius_mean']  # needs a moderate amount of samples
            state_real[:half_length_idx] /= norm_factors['radius_std']  # needs few samples

            # normalize phase
            # heuristic standardization
            # state_real[half_length_idx:] -= norm_factors['phase_mean']  # needs A LOT of samples
            state_real[half_length_idx:] /= norm_factors['phase_std']  # needs few samples

        num_users = satellite_manager.satellites[0].user_nr
        num_antennas = satellite_manager.satellites[0].antenna_nr
        num_satellites = len(satellite_manager.satellites)
        if num_satellites > 1:
            raise ValueError('Not implemented yet, you need to add some math here')  # todo
        keep_indices = np.arange(num_users) * num_antennas
        remove_indices = np.delete(np.arange(num_users * num_antennas), keep_indices)
        state_real = np.delete(state_real, remove_indices)

    elif csi_format == 'real_imag':
        state_real = complex_vector_to_double_real_vector(erroneous_csi)
        if norm_state:

            # state_real *= 1e7  # roughly range [-1, 1]

            # VERY heuristic standardization
            # state_real -= -4.308892163699242e-09
            # state_real /= 7.015404816259004e-08

            # heuristic standardization
            state_real -= norm_factors['mean']
            state_real /= norm_factors['std']

    else:
        raise ValueError(f'Unknown CSI Format {csi_format}')

    return state_real


def get_state_aods(
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_state: bool,
        norm_factors: dict = None,
) -> np.ndarray:
    """TODO: Comment"""

    # FOR CONFIG:
    # self.get_state_args = {
    #     'norm_state': True,  # !!HEURISTIC!!, this will break if you dramatically change the setup
    # }

    if norm_state and norm_factors is None:
        raise ValueError('no norm factors provided')

    state = satellite_manager.get_aods_to_users().flatten()
    if norm_state:

        # heuristic standardization
        state -= norm_factors['mean']
        state /= norm_factors['std']

    return state.flatten()
