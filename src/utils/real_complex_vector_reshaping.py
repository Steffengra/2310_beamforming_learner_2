
"""Provides functions to convert complex to real vectors and the reverse."""

import numpy as np


def complex_vector_to_double_real_vector(
        input_vector: np.ndarray,
) -> np.ndarray:

    return np.concatenate([np.real(input_vector), np.imag(input_vector)])


def real_vector_to_half_complex_vector(
        input_vector: np.ndarray,
) -> np.ndarray:

    real_part_cutoff_index = int(len(input_vector) / 2)

    half_length_complex_vector = input_vector[:real_part_cutoff_index] + 1j * input_vector[real_part_cutoff_index:]

    return half_length_complex_vector


def complex_vector_to_rad_and_phase(
        input_vector: np.ndarray,
) -> np.ndarray:
    """
    Angle is [-pi, pi]
    output: [radius1, radius2, ..., angle1, angle2, ...]
    """

    radius = np.sqrt(np.real(input_vector)**2 + np.imag(input_vector)**2)
    angle = np.arctan2(np.imag(input_vector), np.real(input_vector))

    return np.concatenate([radius, angle])


def rad_and_phase_to_complex_vector(
        input_vector: np.ndarray,
) -> np.ndarray:

    real_part_cutoff_index = int(len(input_vector) / 2)

    half_length_complex_vector = (
            input_vector[:real_part_cutoff_index] * np.cos(input_vector[real_part_cutoff_index:])
            + 1j * input_vector[:real_part_cutoff_index] * np.sin(input_vector[real_part_cutoff_index:])
    )

    return half_length_complex_vector
