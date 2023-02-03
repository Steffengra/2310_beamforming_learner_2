
from datetime import (
    datetime,
)
from pathlib import (
    Path,
)
from numpy import (
    ones,
    infty,
    mean,
    sqrt,
    trace,
    matmul,
)

from src.config.config import (
    Config,
)
from src.data.satellites import (
    Satellites,
)
from src.data.users import (
    Users,
)
from src.models.td3 import (
    TD3ActorCritic,
)
from src.data.channel.los_channel_model import (
    los_channel_model,
)
from src.data.calc_sum_rate import (
    calc_sum_rate,
)
from src.models.get_state import (
    get_state_erroneous_channel_state_information,
)
from src.models.add_random_distribution import (
    add_random_distribution,
)
from src.utils.real_complex_vector_reshaping import (
    real_vector_to_half_complex_vector,
    complex_vector_to_double_real_vector,
)

# TODO: Move this to proper place
import matplotlib.pyplot as plt


def main():

    def progress_print() -> None:
        progress = (
                (training_episode_id * config.config_learner.training_steps_per_episode + training_step_id + 1)
                / (config.config_learner.training_episodes * config.config_learner.training_steps_per_episode)
        )
        timedelta = datetime.now() - real_time_start
        finish_time = real_time_start + timedelta / progress

        print(f'\rSimulation completed: {progress:.2%}, '
              f'est. finish {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}', end='')

    def anneal_parameters() -> tuple:
        if simulation_step > config.config_learner.exploration_noise_step_start_decay:
            exploration_noise_momentum_new = max(
                0.0,
                exploration_noise_momentum - config.config_learner.exploration_noise_linear_decay_per_step
            )
        else:
            exploration_noise_momentum_new = exploration_noise_momentum

        return exploration_noise_momentum_new

    def policy_training_criterion() -> bool:
        """Train policy networks only every k steps and/or only after j total steps to ensure a good value function"""
        if (
            simulation_step > config.config_learner.train_policy_after_j_steps
            and
            (simulation_step % config.config_learner.train_policy_every_k_steps) == 0
        ):
            return True
        return False

    config = Config()
    satellites = Satellites(config=config)
    users = Users(config=config)
    td3 = TD3ActorCritic(rng=config.rng, parent_logger=config.logger, **config.config_learner.td3_args)

    exploration_noise_momentum = config.config_learner.exploration_noise_momentum_initial
    get_state = get_state_erroneous_channel_state_information  # TODO: This might be a config thing

    metrics: dict = {
        'mean_sum_rate_per_episode': -infty * ones(config.config_learner.training_episodes)
    }

    real_time_start = datetime.now()
    if config.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    step_experience: dict = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}

    for training_episode_id in range(config.config_learner.training_episodes):

        episode_metrics: dict = {
            'sum_rate_per_step': -infty * ones(config.config_learner.training_steps_per_episode)
        }

        satellites.calculate_satellite_distances_to_users(users=users.users)
        satellites.calculate_satellite_aods_to_users(users=users.users)
        satellites.calculate_steering_vectors_to_users(users=users.users)
        satellites.update_channel_state_information(channel_model=los_channel_model, users=users.users)
        satellites.update_erroneous_channel_state_information(error_model_config=config.error_model, users=users.users)

        state_next = get_state(satellites=satellites).copy()
        state_next = complex_vector_to_double_real_vector(state_next)

        for training_step_id in range(config.config_learner.training_steps_per_episode):

            simulation_step = training_episode_id * config.config_learner.training_steps_per_episode + training_step_id

            satellites.update_erroneous_channel_state_information(
                error_model_config=config.error_model,
                users=users.users
            )

            # determine state
            state_current = state_next.copy()
            step_experience['state'] = state_current.copy()

            # determine action based on state
            w_precoder_vector = td3.get_action(state=state_current)
            # TODO: Remember that the output of add_random_distribution must be a valid output for
            #  the network too, e.g., be normalized in the same way
            # TODO: At the moment, the policy network output is softmax. that's not necessary, but if
            #  we change it, we also need to change the scale of the exploration noise somehow
            # add exploration noise
            noisy_w_precoder_vector = add_random_distribution(
                rng=config.rng,
                action=w_precoder_vector,
                tau_momentum=exploration_noise_momentum)
            step_experience['action'] = noisy_w_precoder_vector.copy()

            # reshape to fit reward calculation
            noisy_w_precoder_vector = real_vector_to_half_complex_vector(noisy_w_precoder_vector)
            w_precoder = noisy_w_precoder_vector.reshape((config.sat_nr*config.sat_ant_nr, config.user_nr))

            # normalize
            norm_factor = sqrt(config.power_constraint_watt / trace(matmul(w_precoder.conj().T, w_precoder)))
            w_precoder_normalized = norm_factor * w_precoder

            # step simulation based on action, determine reward
            reward = calc_sum_rate(
                channel_state=satellites.channel_state_information,
                w_precoder=w_precoder_normalized,
                noise_power_watt=config.noise_power_watt,
            )
            step_experience['reward'] = reward.copy()

            # update simulation state
            satellites.update_erroneous_channel_state_information(error_model_config=config.error_model, users=users.users)

            # get new state
            state_next = get_state(satellites=satellites).copy()
            state_next = complex_vector_to_double_real_vector(state_next)
            step_experience['next_state'] = state_next

            td3.experience_buffer.add_experience(experience=step_experience)

            # train allocator off-policy
            train_policy = False
            if policy_training_criterion():
                train_policy = True
            td3.train(train_policy=train_policy)

            # anneal parameters
            exploration_noise_momentum = anneal_parameters()

            # log results
            episode_metrics['sum_rate_per_step'][training_step_id] = reward.copy()

            if config.verbosity > 0:
                if training_step_id % 50 == 0:
                    progress_print()

        # log episode results
        metrics['mean_sum_rate_per_episode'][training_episode_id] = mean(episode_metrics['sum_rate_per_step'])
        if config.verbosity == 1:
            print(f' Episode mean reward: {mean(episode_metrics["sum_rate_per_step"])}, current exploration: {exploration_noise_momentum:.2f}')

    # end compute performance profiling
    if config.profile:
        profiler.disable()
        if config.verbosity == 1:
            profiler.print_stats(sort='cumulative')
        profiler.dump_stats(Path(config.performance_profile_path, f'{config.config_learner.training_name}.profile'))

    # TODO: Move this to proper place
    fig, ax = plt.subplots()
    ax.scatter(range(config.config_learner.training_episodes), metrics['mean_sum_rate_per_episode'])
    ax.grid()
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Sum Rate')
    fig.tight_layout()
    if config.show_plots:
        plt.show()


if __name__ == '__main__':
    main()
