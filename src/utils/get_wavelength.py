
from scipy import (
    constants,
)


def get_wavelength(
        freq: float,
) -> float:
    """
    TODO: add description
    """
    wavelength = constants.value('speed of light in vacuum') / freq

    return wavelength
