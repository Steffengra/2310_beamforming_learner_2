
import numpy as np


def spherical_to_cartesian_coordinates(
        spherical_coordinates: np.ndarray,
) -> np.ndarray:
    """
    This function transforms a spherical coordinate vector with (radius,
    inclination, azimuth) into a spherical coordinate vector (x,y,z)
        x = radius * sin(inclination) * cos(azimuth)
        y = radius * sin(inclination) * sin(azimuth)
        z = radius * cos(inclination)
    """

    cartesian_coordinates = np.zeros(3)
    cartesian_coordinates[0] = spherical_coordinates[0] * np.sin(spherical_coordinates[1]) * np.cos(spherical_coordinates[2])
    cartesian_coordinates[1] = spherical_coordinates[0] * np.sin(spherical_coordinates[1]) * np.sin(spherical_coordinates[2])
    cartesian_coordinates[2] = spherical_coordinates[0] * np.cos(spherical_coordinates[1])

    return cartesian_coordinates
