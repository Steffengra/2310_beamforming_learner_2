
from numpy import (
    ndarray,
    concatenate,
    real,
    imag,
    sqrt,
    sin,
    cos,
    arctan2,
)


def complex_vector_to_double_real_vector(
        input_vector: ndarray,
) -> ndarray:

    return concatenate([real(input_vector), imag(input_vector)],
                       dtype='float32')


def real_vector_to_half_complex_vector(
        input_vector: ndarray,
) -> ndarray:

    real_part_cutoff_index = int(len(input_vector) / 2)

    half_length_complex_vector = input_vector[:real_part_cutoff_index] + 1j * input_vector[real_part_cutoff_index:]

    return half_length_complex_vector


def complex_vector_to_rad_and_phase(
        input_vector: ndarray,
) -> ndarray:
    """
    Angle is [-pi, pi]
    """

    radius = sqrt(real(input_vector)**2 + imag(input_vector)**2)
    angle = arctan2(imag(input_vector), real(input_vector))

    return concatenate([radius, angle])


def rad_and_phase_to_complex_vector(
        input_vector: ndarray,
) -> ndarray:

    real_part_cutoff_index = int(len(input_vector) / 2)

    half_length_complex_vector = (
            input_vector[:real_part_cutoff_index] * cos(input_vector[real_part_cutoff_index:])
            + 1j * input_vector[:real_part_cutoff_index] * sin(input_vector[real_part_cutoff_index:])
    )

    return half_length_complex_vector
