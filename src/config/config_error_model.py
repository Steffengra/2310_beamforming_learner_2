
from src.data.channel.los_channel_error_model_no_error import (
    los_channel_error_model_no_error,
)
from src.data.channel.los_channel_error_model_multiplicative_on_cos import (
    los_channel_error_model_multiplicative_on_cos,
)
from src.data.channel.los_channel_error_model_in_sat2user_dist import (
    los_channel_error_model_in_sat2user_dist,
)


class ConfigErrorModel:
    """
    Defines parameters for the error model
    """

    def __init__(
            self,
    ) -> None:

        # self.error_model = los_channel_error_model_no_error
        self.error_model = los_channel_error_model_multiplicative_on_cos
        # self.error_model = los_channel_error_model_in_sat2user_dist

        self.update()

    def _set_params(
            self,
    ) -> None:

        # NO ERROR MODEL
        #  This is a dummy error model
        if self.error_model == los_channel_error_model_no_error:
            self.error_model_name = 'err_no'

        # MULTIPLICATIVE ERROR MODEL
        #  In this case, the error is not directly added to the AODs but uniformly
        #  distributed on the cos(aods)
        if self.error_model == los_channel_error_model_multiplicative_on_cos:
            self.error_model_name: str = 'err_mult_on_steering_cos'
            self.uniform_error_interval: dict = {
                'low': -0.0,
                'high': 0.0,
            }

        # SAT2USER DISTANCE ERROR MODEL
        #  This error model calculates an erroneous channel state information estimate based on a
        #  perturbed satellite to user distance estimate.
        if self.error_model == los_channel_error_model_in_sat2user_dist:
            self.error_model_name: str = 'err_sat2userdist'
            self.distance_error_std: float = 0/100_000_000  # zB 1/100_000_000, 2/100_000_000..

        # Normal distributed directly on AODs ??? TODO

    def update(self):
        self._set_params()
