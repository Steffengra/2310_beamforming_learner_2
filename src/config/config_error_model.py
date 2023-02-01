
from src.data.channel.los_channel_error_model_multiplicative_on_cos import (
    los_channel_error_model_multiplicative_on_cos,
)


class ConfigErrorModel:
    """
    Defines parameters for the error model
    """

    def __init__(
            self,
    ) -> None:

        self.iter_nr: int = 10_000  # Number of Monte Carlo iterations

        self.error_model = los_channel_error_model_multiplicative_on_cos

        # MULTIPLICATIVE ERROR MODEL
        #  In this case, the error is not directly added to the AODs but uniformly
        #  distributed on the cos(aods)
        if self.error_model == los_channel_error_model_multiplicative_on_cos:
            self.uniform_error_interval: dict = {
                'low': 0,
                'high': 0,
            }

        # Normal distributed directly on AODs ??? TODO
