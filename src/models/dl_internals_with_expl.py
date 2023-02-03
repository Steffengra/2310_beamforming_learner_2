
from tensorflow.keras.optimizers import (
    Adam,
    Nadam
)

from tensorflow import (
    Tensor,
    where,
    abs as tf_abs,
    subtract as tf_subtract,
    multiply as tf_multiply,
    square as tf_square,
    reduce_mean as tf_reduce_mean
)


def optimizer_adam(
        learning_rate: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False

):
    """
    Adam (2015) tries to combine advantages of AdaGrad (2010), RMSProp (2012).

    Further small improvements, e.g., bias correction etc.

    Adam adaptively changes learning rate per parameter based on
    1st and 2nd order estimates of the parameters gradient.
    To do this, it calculates an exponentially decaying
    moving average of grad and grad^2, with exponential
    decay factors beta_1 (1st) and beta_2 (2nd).

    Update equation:
        x_t = x_t-1 - lr * grad_mean_hat_t / sqrt(grad_var_hat_t)
        with
        grad_mean_hat_t = beta_1 * grad_mean_hat_t-1 + (1 - beta_1) * grad_t
        grad_var_hat_t  = beta_2 * grad_var_hat_t-1 + (1 - beta_2) * grad_t^2

    Amsgrad is a 2018 variant of Adam. It identifies the exponentially
    decaying moving average as a source of error in convergence
    behaviour and proposes to fix this by keeping the maximum var estimate:
        x_t = x_t-1 - lr * grad_mean_hat_t / sqrt(max_grad_var_hat_t)
        with
        max_grad_var_hat_t = max(grad_var_hat_t, max_grad_var_hat_t-1)

    Suggested alternative is a sgd with nesterov momentum,
    e.g., NAdam (2016), Vanilla+nesterov

    Args:
        learning_rate: The sgd step size
        beta_1: beta_1 is the exponential decay for grad. mean estimate
                "beta_1 works similar to momentum in other sgd"
        beta_2: beta_2 is the exponential decay for grad. variance estimate
                "A beta_2 close to 1 helps with sparse gradients"
        epsilon: epsilon helps with numerical stability
        amsgrad: AMSgrad is a Variant of Adam
    """

    optimizer = Adam(
        learning_rate=learning_rate,  # default 1e-3
        beta_1=beta_1,  # default 0.9
        beta_2=beta_2,  # default 0.999
        epsilon=epsilon,  # default 1e-7. tf docs suggest up to 0.1 or 1? Strange
        amsgrad=amsgrad  # default=False
    )

    return optimizer


def optimizer_nadam(
        learning_rate: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7
):
    """
    [1] Incorporating Nesterov Momentum into Adam, Dozat 2015
    NAdam implements Adam with Nesterov momentum.
    For details on Adam see optimizer_adam

    Nesterov Momentum applies the momentum grad estimate
    before calculating the next grad, i.e.,
    it calculates the gradient not on f(theta_t-1),
    but on f(theta_t-1 - lr * beta_1 * grad_mean_estimate_t-1).
    This results in empirically faster convergence and
    some provably better characteristics [1].

    Args:
        learning_rate: The sgd step size
        beta_1: beta_1 is the exponential decay for grad. mean estimate
                "beta_1 works similar to momentum in other sgd"
        beta_2: beta_2 is the exponential decay for grad. variance estimate
                "A beta_2 close to 1 helps with sparse gradients"
        epsilon: epsilon helps with numerical stability
    """

    optimizer = Nadam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon
    )

    return optimizer


def mse_loss(
        td_error,
        importance_sampling_weights
) -> Tensor:
    squared_error = tf_multiply(importance_sampling_weights, tf_square(td_error))

    return tf_reduce_mean(squared_error)


def huber_loss(
        td_error,
        importance_sampling_weights
) -> Tensor:
    """
    Huber loss fixes the grad magnitude to at most 1 by linearizing the loss curve after f'(x)=1.

    Huber loss tries to prevent 'exploding gradients'.
    """
    delta = 1  # Where to clip

    # delta * (abs(td) - .5 * delta)
    absolute_error = tf_multiply(
        delta,
        tf_subtract(
            tf_abs(td_error),
            tf_multiply(.5, delta)
        )
    )

    squared_error = tf_multiply(
        .5,
        tf_square(td_error)
    )

    indicator = absolute_error < delta
    loss = where(indicator, squared_error, absolute_error)

    return tf_reduce_mean(importance_sampling_weights * loss)
