
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

        #self.error_model = los_channel_error_model_multiplicative_on_cos
        self.error_model = los_channel_error_model_in_sat2user_dist

        # MULTIPLICATIVE ERROR MODEL
        #  In this case, the error is not directly added to the AODs but uniformly
        #  distributed on the cos(aods)
        if self.error_model == los_channel_error_model_multiplicative_on_cos:
            self.uniform_error_interval: dict = {
                'low': 0,
                'high': 0,
            }

        # SAT2USER DISTANCE ERROR MODEL
        # This error model models unknown phase shifts between satellites TODO
        if self.error_model == los_channel_error_model_in_sat2user_dist:
            self.distance_error_std: float = 1/100_000_000  # zB 1/100_000_000, 2/100_000_000..

        # Normal distributed directly on AODs ??? TODO
