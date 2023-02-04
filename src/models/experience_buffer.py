
from numpy import (
    ndarray,
    zeros,
    sum as np_sum,
    divide as np_divide,
    power as np_power,
    max as np_max,
    where as np_where,
    count_nonzero as np_count_nonzero,
)
from numpy.random import default_rng


class ExperienceBuffer:
    def __init__(
            self,
            rng: default_rng,
            buffer_size: int,
            priority_scale_alpha: float,  # alpha=0 is uniform sampling, alpha=1 is fully prioritized sampling
            importance_sampling_correction_beta: float,  # beta=1 is full correction, beta=0 is no correction
    ) -> None:
        self.rng: default_rng = rng
        self.write_pointer: int = 0

        self.buffer_size = buffer_size
        self.buffer: list = [{}] * self.buffer_size
        self.priorities: ndarray = zeros(self.buffer_size, dtype='float32')
        self.probabilities: ndarray = zeros(self.buffer_size, dtype='float32')  # prob_i = prio_i / sum_i(prio_i)

        self.priority_scale_alpha: float = priority_scale_alpha
        self.importance_sampling_correction_beta: float = importance_sampling_correction_beta
        self.min_priority: float = 1e-20
        self.max_priority: float = self.min_priority

    def get_len(self) -> int:
        return np_count_nonzero(self.priorities)

    def add_experience(
            self,
            experience: dict,
    ) -> None:
        self.buffer[self.write_pointer] = experience.copy()
        self.priorities[self.write_pointer] = self.max_priority

        self.write_pointer += 1
        self.write_pointer = self.write_pointer % self.buffer_size

    def sample(
            self,
            batch_size: int,
    ) -> tuple[list, ndarray, ndarray]:
        # Update Probabilities
        priority_sum = np_sum(self.priorities)
        self.probabilities = np_divide(self.priorities, priority_sum, dtype='float32')

        # Sample
        sample_experience_ids = self.rng.choice(
            a=self.buffer_size,
            size=batch_size,
            replace=True,  # Can an experience id be selected multiple times? Yes/No
            p=self.probabilities,
        )

        sample_experiences = [self.buffer[ii] for ii in sample_experience_ids]
        sample_probabilities = self.probabilities[sample_experience_ids]

        sample_importance_weights = np_power(sample_probabilities,
                                             -self.importance_sampling_correction_beta, dtype='float32')
        sample_importance_weights = np_divide(sample_importance_weights,
                                              np_max(sample_importance_weights), dtype='float32')

        return (
            sample_experiences,
            sample_experience_ids,
            sample_importance_weights,
        )

    def adjust_priorities(
            self,
            experience_ids: ndarray,
            new_priorities: ndarray,
    ) -> None:
        new_priorities = np_power(new_priorities, self.priority_scale_alpha, dtype='float32')
        self.priorities[experience_ids] = np_where(new_priorities > self.min_priority,
                                                   new_priorities, self.min_priority)

        sample_max_priority = np_max(new_priorities)
        if sample_max_priority > self.max_priority:
            self.max_priority = sample_max_priority
