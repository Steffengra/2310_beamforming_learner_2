
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

    # FOR CONFIG:
    # self.get_state_args = {
    #     'csi_format': 'rad_phase',  # 'rad_phase', 'real_imag'
    #     'norm_csi': True,  # !!HEURISTIC!!, this will break if you dramatically change the setup
    # }

    erroneous_csi = satellites.erroneous_channel_state_information.flatten()

    if csi_format == 'rad_phase':
        state_real = complex_vector_to_rad_and_phase(erroneous_csi)
        if norm_csi:
            half_length_idx = int(len(state_real) / 2)
            state_real[:half_length_idx] = state_real[:half_length_idx] - 1.4057e-7  # due to high distance, these digits will be roughly the same for all csi
            state_real[:half_length_idx] = state_real[:half_length_idx] * 1e11  # roughly [0, 1]

            state_real[half_length_idx:] = state_real[half_length_idx:] / pi  # [-1, 1]
            state_real[half_length_idx:] = state_real[half_length_idx:] + 1  # [0, 2]
            state_real[half_length_idx:] = state_real[half_length_idx:] / 2  # [0, 1]

    elif csi_format == 'real_imag':
        state_real = complex_vector_to_double_real_vector(erroneous_csi)
        if norm_csi:
            state_real *= 1e7  # roughly range -1 ... 1
            state_real += 1  # roughly range 0 ... 2
            state_real /= 2  # roughly range 0 ... 1

    else:
        raise ValueError(f'Unknown CSI Format {csi_format}')

    return state_real


def get_state_aods(
        satellites,
        norm_aods,
) -> ndarray:

    # FOR CONFIG:
    # self.get_state_args = {
    #     'norm_aods': True,  # !!HEURISTIC!!, this will break if you dramatically change the setup
    # }

    state = satellites.get_aods_to_users().flatten()
    if norm_aods:
        state -= pi/2
        state *= 1e2  # very roughly [-2, 2]

    return state.flatten()
