
from numpy import (
    ndarray,
    pi,
)

from src.utils.real_complex_vector_reshaping import (
    real_vector_to_half_complex_vector,
    complex_vector_to_double_real_vector,
    rad_and_phase_to_complex_vector,
    complex_vector_to_rad_and_phase,
)


# TODO: Norm values hardcoded can cause problems with different applications

def get_state_erroneous_channel_state_information(
        satellites,
        csi_format: str,
        norm_csi: bool,
) -> ndarray:

    erroneous_csi = satellites.erroneous_channel_state_information.flatten()

    if csi_format == 'rad_phase':
        state_real = complex_vector_to_rad_and_phase(erroneous_csi)
        if norm_csi:
            half_length_idx = int(len(state_real) / 2)
            state_real[:half_length_idx] = state_real[:half_length_idx] * 1e7
            state_real[half_length_idx:] = state_real[half_length_idx:] / pi

    elif csi_format == 'real_imag':
        state_real = complex_vector_to_double_real_vector(erroneous_csi)
        if norm_csi:
            state_real *= 1e8

    else:
        raise ValueError(f'Unknown CSI Format {csi_format}')

    return state_real


def get_state_aods(
        satellites,
) -> ndarray:

    state = satellites.get_aods_to_users()

    return state.flatten()
