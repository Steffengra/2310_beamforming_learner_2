
from numpy import (
    ndarray,
    concatenate,
    real,
    imag,
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
