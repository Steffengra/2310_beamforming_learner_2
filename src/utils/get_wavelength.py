
from scipy import constants


def get_wavelength(
        freq: float,
) -> float:
    """Calculates the wavelength from frequency."""

    wavelength = constants.value('speed of light in vacuum') / freq

    return wavelength
