
from numpy import (
    ndarray,
)

from src.data.precoder.mmse_precoder import (
    mmse_precoder_no_norm,
)
from src.utils.real_complex_vector_reshaping import (
    complex_vector_to_double_real_vector,
)


# TODO: Remember that the output of this function must
#  be a valid output from the policy network, e.g., be normalized in the same way
def add_random_distribution(
        rng,
        action: ndarray,
        tau_momentum: float,
) -> ndarray:
    """
    Mix an action vector with a random_uniform vector of same length
    by tau * random_distribution + (1 - tau) * action
    """

    if tau_momentum == 0.0:
        return action

    # create random action
    # random_distribution = rng.random(size=len(action), dtype='float32')
    # random_distribution = random_distribution / sum(random_distribution)
    # TODO: These values are taken from the mmse precoder, but probably shouldnt be hardcoded
    random_distribution = rng.normal(loc=0.03353971, scale=0.5, size=len(action))

    # combine
    noisy_action = tau_momentum * random_distribution + (1 - tau_momentum) * action

    # normalize
    # sum_noisy_action = sum(noisy_action)
    # if sum_noisy_action != 0:
    #     noisy_action = noisy_action / sum_noisy_action

    return noisy_action


def add_mmse_precoder(
        action: ndarray,
        tau_momentum: float,
        channel_matrix,
        noise_power_watt,
        power_constraint_watt,
) -> ndarray:

    if tau_momentum == 0.0:
        return action

    w_mmse = mmse_precoder_no_norm(
        channel_matrix=channel_matrix,
        noise_power_watt=noise_power_watt,
        power_constraint_watt=power_constraint_watt,
    )
    w_mmse_real = complex_vector_to_double_real_vector(w_mmse)

    noisy_action = tau_momentum * w_mmse_real + (1 - tau_momentum) * action

    return noisy_action
