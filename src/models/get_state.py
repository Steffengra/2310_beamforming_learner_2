
from numpy import (
    ndarray,
)

from src.data.satellites import (
    Satellites,
)


def get_state_erroneous_channel_state_information(
        satellites: Satellites,
) -> ndarray:

    state = satellites.erroneous_channel_state_information
    state = state.flatten()

    return state


def get_state_aods(
        satellites: Satellites,
) -> ndarray:

    state = satellites.get_aods_to_users()

    return state.flatten()
