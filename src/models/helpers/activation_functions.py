
from tensorflow import (
    Tensor,
    tanh,
    multiply,
    less,
    where
)


def activation_penalized_tanh(
        x: Tensor
) -> Tensor:
    y = tanh(x)
    y = where(less(x, 0), multiply(y, .25), y)

    return y
