
"""
These functions can be used to create "noisy actions", thereby forcing exploration.
TODO: Remember that the output of this function must
    be a valid output from the policy network, e.g., be normalized in the same way
"""

import numpy as np

from src.data.precoder.mmse_precoder import (
    mmse_precoder_no_norm,
)
from src.utils.real_complex_vector_reshaping import (
    complex_vector_to_double_real_vector,
)


def add_random_distribution(
        rng: np.random.default_rng,
        action: np.ndarray,
        tau_momentum: float,
        normalize: bool = False,
) -> np.ndarray:
    """
    Mix an action vector with a random_uniform vector of same length
    by tau * random_distribution + (1 - tau) * action.
    """

    if tau_momentum == 0.0:
        return action

    # create random action
    random_distribution = rng.normal(loc=np.mean(action), scale=np.std(action))

    # combine
    noisy_action = tau_momentum * random_distribution + (1 - tau_momentum) * action

    # normalize
    if normalize:
        sum_noisy_action = sum(noisy_action)
        if sum_noisy_action != 0:
            noisy_action = noisy_action / sum_noisy_action

    return noisy_action


def add_mmse_precoder(
        action: np.ndarray,
        tau_momentum: float,
        channel_matrix: np.ndarray,
        noise_power_watt: float,
        power_constraint_watt: float,
) -> np.ndarray:

    if tau_momentum == 0.0:
        return action

    w_mmse = mmse_precoder_no_norm(
        channel_matrix=channel_matrix,
        noise_power_watt=noise_power_watt,
        power_constraint_watt=power_constraint_watt,
    )
    w_mmse = w_mmse.flatten()
    w_mmse_real = complex_vector_to_double_real_vector(w_mmse)

    noisy_action = tau_momentum * w_mmse_real + (1 - tau_momentum) * action

    return noisy_action
