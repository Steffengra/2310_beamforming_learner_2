
import numpy as np


class ExperienceBuffer:
    """The circular Experience Buffer object holds a number of objects, e.g., dicts, and can be sampled."""

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
        self.min_priority: float = 1e-12
        self.max_priority: float = self.min_priority

    def get_len(self) -> int:
        return np.count_nonzero(self.priorities)

    def add_experience(
            self,
            experience: dict,
    ) -> None:
        """Add an object to the current position of the buffer."""

        self.buffer[self.write_pointer] = experience.copy()
        self.priorities[self.write_pointer] = self.max_priority

        self.write_pointer += 1
        self.write_pointer = self.write_pointer % self.buffer_size

    def sample(
            self,
            batch_size: int,
    ) -> tuple[list, np.ndarray, np.ndarray]:
        """Sample a number of objects from the buffer. Objects can have different sampling probablities."""

        # Update Probabilities
        priority_sum = np.sum(self.priorities)
        self.probabilities = np.divide(self.priorities, priority_sum)

        # Sample
        if self.priority_scale_alpha != 0:
            sample_experience_ids = self.rng.choice(
                a=self.buffer_size,
                size=batch_size,
                replace=False,  # Can an experience id be selected multiple times? Yes/No
                p=self.probabilities,
            )
        else:
            if any(self.probabilities == 0):
                sample_experience_ids = self.rng.choice(
                    a=self.write_pointer,
                    size=batch_size,
                    replace=False,  # Can an experience id be selected multiple times? Yes/No
                )
            else:
                sample_experience_ids = self.rng.choice(
                    a=self.buffer_size,
                    size=batch_size,
                    replace=False,  # Can an experience id be selected multiple times? Yes/No
                )

        sample_experiences = [self.buffer[ii] for ii in sample_experience_ids]
        sample_probabilities = self.probabilities[sample_experience_ids]

        if self.importance_sampling_correction_beta != 1:
            sample_importance_weights = np.power(sample_probabilities,
                                                 -self.importance_sampling_correction_beta)
            sample_importance_weights = np.divide(sample_importance_weights,
                                                  np.max(sample_importance_weights))
        else:
            sample_importance_weights = sample_probabilities / np.max(sample_probabilities)

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
        """Adjust the priority -> sampling probability of a number of objects in buffer."""

        if self.priority_scale_alpha != 0:
            new_priorities = np.power(new_priorities, self.priority_scale_alpha)
            self.priorities[experience_ids] = np.where(new_priorities > self.min_priority,
                                                       new_priorities, self.min_priority)
            sample_max_priority = np.max(new_priorities)

        else:
            self.priorities[experience_ids] = 1
            sample_max_priority = 1

        if sample_max_priority > self.max_priority:
            self.max_priority = sample_max_priority

    def clear(
            self,
    ) -> None:
        """Reset the buffer."""

        self.write_pointer: int = 0

        self.buffer: list = [{}] * self.buffer_size
        self.priorities: np.ndarray = np.zeros(self.buffer_size)
        self.probabilities: np.ndarray = np.zeros(self.buffer_size)  # prob_i = prio_i / sum_i(prio_i)

        self.max_priority: float = self.min_priority
