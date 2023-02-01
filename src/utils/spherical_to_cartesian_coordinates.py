
from numpy import (
    zeros,
    sin,
    cos,
)


def spherical_to_cartesian_coordinates(spherical_coordinates):
    """
    This function transforms a spherical coordinate vector with (radius,
    inclination, azimuth) into a spherical coordinate vector (x,y,z)
        x = radius * sin(inclination) * cos(azimuth)
        y = radius * sin(inclination) * sin(azimuth)
        z = radius * cos(inclination)
    """
    cartesian_coordinates = zeros(3)
    cartesian_coordinates[0] = spherical_coordinates[0] * sin(spherical_coordinates[1]) * cos(spherical_coordinates[2])
    cartesian_coordinates[1] = spherical_coordinates[0] * sin(spherical_coordinates[1]) * sin(spherical_coordinates[2])
    cartesian_coordinates[2] = spherical_coordinates[0] * cos(spherical_coordinates[1])

    return cartesian_coordinates
