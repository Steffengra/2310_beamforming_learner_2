
from numpy import (
    ndarray,
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
    random_distribution = rng.random(size=len(action), dtype='float32')
    random_distribution = random_distribution / sum(random_distribution)

    # combine
    noisy_action = tau_momentum * random_distribution + (1 - tau_momentum) * action

    # normalize
    sum_noisy_action = sum(noisy_action)
    if sum_noisy_action != 0:
        noisy_action = noisy_action / sum_noisy_action

    return noisy_action
