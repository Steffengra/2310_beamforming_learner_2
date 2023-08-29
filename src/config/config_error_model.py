
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
            user_nr: int,
    ) -> None:

        self.rng = rng

        self.error_rng_parametrizations: dict = {}

        self._wavelength = wavelength
        self._user_nr = user_nr

        if channel_model == los_channel_model:
            self.set_los_channel_error_parameters()
            self.error_rngs = self.set_los_channel_error_functions()

        else:
            raise ValueError(f'Unknown channel model {channel_model}')

    def set_los_channel_error_parameters(
            self,
    ) -> None:

        self.error_rng_parametrizations['satellite_to_user_distance_error'] = {
            'distribution': self.rng.uniform,
            'args': {
                'low': 0,
                'high': 0,
                'size': self._user_nr,
            },
        }
        self.error_rng_parametrizations['additive_error_on_aod'] = {
            'distribution': self.rng.normal,
            'args': {
                'loc': 0,
                'scale': 0,
                'size': self._user_nr,
            },
        }
        self.error_rng_parametrizations['additive_error_on_cosine_of_aod'] = {
            'distribution': self.rng.uniform,
            'args': {
                'low': -0.0,
                'high': 0.0,
                'size': self._user_nr,
            },
        }

    def set_los_channel_error_functions(
            self,
    ) -> dict:

        def roll_additive_error_on_overall_phase_shift():
            roll_satellite_to_user_distance_error = self.error_rng_parametrizations['satellite_to_user_distance_error']['distribution'](
                **self.error_rng_parametrizations['satellite_to_user_distance_error']['args']
            )
            return 2 * np.pi / self._wavelength * (roll_satellite_to_user_distance_error % self._wavelength)

        def roll_additive_error_on_aod():
            return self.error_rng_parametrizations['additive_error_on_aod']['distribution'](
                **self.error_rng_parametrizations['additive_error_on_aod']['args']
            )

        def roll_additive_error_on_cosine_of_aod():
            return self.error_rng_parametrizations['additive_error_on_cosine_of_aod']['distribution'](
                **self.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']
            )

        def roll_additive_error_on_channel_vector():
            return np.zeros(1)

        error_rngs = {
            'additive_error_on_overall_phase_shift': roll_additive_error_on_overall_phase_shift,
            'additive_error_on_aod': roll_additive_error_on_aod,
            'additive_error_on_cosine_of_aod': roll_additive_error_on_cosine_of_aod,
            'additive_error_on_channel_vector': roll_additive_error_on_channel_vector,
        }

        return error_rngs

    def set_zero_error(
            self,
    ) -> None:

        for parameter_name, parameter_content in self.error_rng_parametrizations.items():
            for arg, value in parameter_content['args'].items():
                if arg != 'size':
                    self.error_rng_parametrizations[parameter_name]['args'][arg] = 0

        self.set_los_channel_error_functions()
