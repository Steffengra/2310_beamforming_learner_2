
import numpy as np


class ExperienceBuffer:
    def __init__(
            self,
            rng: np.random.Generator,
            buffer_size: int,
            priority_scale_alpha: float,  # alpha=0 is uniform sampling, alpha=1 is fully prioritized sampling
            importance_sampling_correction_beta: float,  # beta=1 is full correction, beta=0 is no correction
    ) -> None:
        self.rng: np.random.Generator = rng
        self.write_pointer: int = 0

        self.buffer_size = buffer_size
        self.buffer: list = [{}] * self.buffer_size
        self.priorities: np.ndarray = np.zeros(self.buffer_size)
        self.probabilities: np.ndarray = np.zeros(self.buffer_size)  # prob_i = prio_i / sum_i(prio_i)

        self.priority_scale_alpha: float = priority_scale_alpha
        self.importance_sampling_correction_beta: float = importance_sampling_correction_beta
        self.min_priority: float = 1e-20
        self.max_priority: float = self.min_priority

    def get_len(self) -> int:
        return np.count_nonzero(self.priorities)

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
    ) -> tuple[list, np.ndarray, np.ndarray]:

        # Update Probabilities
        priority_sum = np.sum(self.priorities)
        self.probabilities = np.divide(self.priorities, priority_sum)

        # Sample
        sample_experience_ids = self.rng.choice(
            a=self.buffer_size,
            size=batch_size,
            replace=False,  # Can an experience id be selected multiple times? Yes/No
            p=self.probabilities,
        )

        sample_experiences = [self.buffer[ii] for ii in sample_experience_ids]
        sample_probabilities = self.probabilities[sample_experience_ids]

        sample_importance_weights = np.power(sample_probabilities,
                                             -self.importance_sampling_correction_beta)
        sample_importance_weights = np.divide(sample_importance_weights,
                                              np.max(sample_importance_weights))

        return (
            sample_experiences,
            sample_experience_ids,
            sample_importance_weights,
        )

    def adjust_priorities(
            self,
            experience_ids: np.ndarray,
            new_priorities: np.ndarray,
    ) -> None:

        new_priorities = np.power(new_priorities, self.priority_scale_alpha)
        self.priorities[experience_ids] = np.where(new_priorities > self.min_priority,
                                                   new_priorities, self.min_priority)

        sample_max_priority = np.max(new_priorities)
        if sample_max_priority > self.max_priority:
            self.max_priority = sample_max_priority

    def clear(
            self,
    ) -> None:

        self.write_pointer: int = 0

        self.buffer: list = [{}] * self.buffer_size
        self.priorities: np.ndarray = np.zeros(self.buffer_size)
        self.probabilities: np.ndarray = np.zeros(self.buffer_size)  # prob_i = prio_i / sum_i(prio_i)

        self.max_priority: float = self.min_priority
