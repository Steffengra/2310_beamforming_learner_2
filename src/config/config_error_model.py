
import numpy as np

from src.data.channel.los_channel_model import los_channel_model


class ConfigErrorModel:
    """
    Defines parameters for the error model
    """

    def __init__(
            self,
            channel_model,
            rng: np.random.Generator,
            wavelength: float,
    ) -> None:

        self.rng = rng

        if channel_model == los_channel_model:
            self.error_rngs = self.set_los_channel_errors(
                wavelength=wavelength,
            )

        else:
            raise ValueError(f'Unknown channel model {channel_model}')

    def set_los_channel_errors(
            self,
            wavelength: float,
    ) -> dict:

        def roll_additive_error_on_overall_phase_shift():
            roll_satellite_to_user_distance_error = self.rng.uniform(-5, 5, size=None)
            return 2 * np.pi / wavelength * (roll_satellite_to_user_distance_error % wavelength)

        def roll_additive_error_on_aod():
            return self.rng.normal(0, 1, size=None)

        def roll_additive_error_on_cosine_of_aod():
            return self.rng.uniform(-5, 5, size=None)

        def roll_additive_error_on_channel_vector():
            return np.zeros(1)

        error_rngs = {
            'additive_error_on_overall_phase_shift': roll_additive_error_on_overall_phase_shift,
            'additive_error_on_aod': roll_additive_error_on_aod,
            'additive_error_on_cosine_of_aod': roll_additive_error_on_cosine_of_aod,
            'additive_error_on_channel_vector': roll_additive_error_on_channel_vector,
        }

        return error_rngs
