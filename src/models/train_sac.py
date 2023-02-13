
from datetime import (
    datetime,
)
from pathlib import (
    Path,
)
from shutil import (
    copytree,
)
from gzip import (
    open as gzip_open,
)
from pickle import (
    dump as pickle_dump,
)
from numpy import (
    ones,
    infty,
    mean,
    std,
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
from src.models.algorithms.soft_actor_critic import (
    SoftActorCritic,
)
from src.data.channel.los_channel_model import (
    los_channel_model,
)
from src.data.calc_sum_rate import (
    calc_sum_rate,
)
from src.data.precoder.mmse_precoder import (
    mmse_precoder_normalized,
)
from src.utils.real_complex_vector_reshaping import (
    real_vector_to_half_complex_vector,
    complex_vector_to_double_real_vector,
    complex_vector_to_rad_and_phase,
)
from src.utils.norm_precoder import (
    norm_precoder,
)
from src.utils.plot_sweep import (
    plot_sweep,
)

# TODO: Move this to proper place
import matplotlib.pyplot as plt


def train_sac_single_error(config):

    def progress_print() -> None:
        progress = (
                (training_episode_id * config.config_learner.training_steps_per_episode + training_step_id + 1)
                / (config.config_learner.training_episodes * config.config_learner.training_steps_per_episode)
        )
        timedelta = datetime.now() - real_time_start
        finish_time = real_time_start + timedelta / progress

        print(f'\rSimulation completed: {progress:.2%}, '
              f'est. finish {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}', end='')

    def policy_training_criterion() -> bool:
        """Train policy networks only every k steps and/or only after j total steps to ensure a good value function"""
        if (
            simulation_step > config.config_learner.train_policy_after_j_steps
            and
            (simulation_step % config.config_learner.train_policy_every_k_steps) == 0
        ):
            return True
        return False

    def sim_update():
        users.update_positions(config=config)
        satellites.update_positions(config=config)

        satellites.calculate_satellite_distances_to_users(users=users.users)
        satellites.calculate_satellite_aods_to_users(users=users.users)
        satellites.calculate_steering_vectors_to_users(users=users.users)
        satellites.update_channel_state_information(channel_model=los_channel_model, users=users.users)
        satellites.update_erroneous_channel_state_information(error_model_config=config.error_model, users=users.users)

    def add_mmse_experience():
        w_mmse = mmse_precoder_normalized(
            channel_matrix=satellites.erroneous_channel_state_information,
            **config.mmse_args
        ).flatten()
        reward_mmse = calc_sum_rate(
            channel_state=satellites.channel_state_information,
            w_precoder=w_precoder_normed,
            noise_power_watt=config.noise_power_watt,
        )
        mmse_experience = {
            'state': state_current,
            'action': complex_vector_to_double_real_vector(w_mmse),
            'reward': reward_mmse,
            'state_next': state_next,
        }
        sac.add_experience(mmse_experience)

    def save_model_checkpoint(extra=''):

        name = f'error_{config.error_model.uniform_error_interval["high"]}_userwiggle_{config.user_dist_variance}'
        if extra != '':
            name += f'_snapshot_{extra:.2f}'
        sac.networks['policy'][0]['primary'].save(
            Path(
                config.trained_models_path,
                config.config_learner.training_name,
                'single_error',
                name,
                'model'
            )
        )

        # save config
        copytree(Path(config.project_root_path, 'src', 'config'),
                 Path(config.project_root_path, 'models', config.config_learner.training_name, 'single_error', name,
                      'config'),
                 dirs_exist_ok=True)

    def save_results():

        name = f'training_error_{config.error_model.uniform_error_interval["high"]}_userwiggle_{config.user_dist_variance}.gzip'
        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'single_error')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip_open(Path(results_path, name), 'wb') as file:
            pickle_dump(metrics, file=file)

    satellites = Satellites(config=config)
    users = Users(config=config)
    sac = SoftActorCritic(rng=config.rng, **config.config_learner.algorithm_args)

    metrics: dict = {
        'mean_sum_rate_per_episode': -infty * ones(config.config_learner.training_episodes)
    }
    high_score = -infty

    real_time_start = datetime.now()
    if config.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    step_experience: dict = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}

    for training_episode_id in range(config.config_learner.training_episodes):

        episode_metrics: dict = {
            'sum_rate_per_step': -infty * ones(config.config_learner.training_steps_per_episode),
            'mean_log_prob_density': infty * ones(config.config_learner.training_steps_per_episode),
            'value_loss': -infty * ones(config.config_learner.training_steps_per_episode),
        }

        sim_update()

        state_next = config.config_learner.get_state(satellites=satellites, **config.config_learner.get_state_args)

        for training_step_id in range(config.config_learner.training_steps_per_episode):

            simulation_step = training_episode_id * config.config_learner.training_steps_per_episode + training_step_id

            # determine state
            state_current = state_next
            step_experience['state'] = state_current

            # determine action based on state
            action = sac.get_action(state=state_current)
            step_experience['action'] = action

            # reshape to fit reward calculation
            w_precoder_vector = real_vector_to_half_complex_vector(action)
            w_precoder = w_precoder_vector.reshape((config.sat_nr*config.sat_ant_nr, config.user_nr))
            w_precoder_normed = norm_precoder(precoding_matrix=w_precoder, power_constraint_watt=config.power_constraint_watt,
                                              per_satellite=True, sat_nr=config.sat_nr, sat_ant_nr=config.sat_ant_nr)

            # step simulation based on action, determine reward
            reward = calc_sum_rate(
                channel_state=satellites.channel_state_information,
                w_precoder=w_precoder_normed,
                noise_power_watt=config.noise_power_watt,
            )
            step_experience['reward'] = reward

            # optionally add the corresponding mmse precoder to the data set
            if config.rng.random() < 0.0:
                add_mmse_experience()  # todo note: currently state_next saved in the mmse experience is not correct

            # update simulation state
            sim_update()

            # get new state
            state_next = config.config_learner.get_state(satellites=satellites, **config.config_learner.get_state_args)
            step_experience['next_state'] = state_next

            sac.add_experience(experience=step_experience)

            # train allocator off-policy
            train_policy = False
            if policy_training_criterion():
                train_policy = True
            mean_log_prob_density, value_loss = sac.train(
                toggle_train_value_networks=True,
                toggle_train_policy_network=train_policy,
                toggle_train_entropy_scale_alpha=True,
            )

            # log results
            episode_metrics['sum_rate_per_step'][training_step_id] = reward
            episode_metrics['mean_log_prob_density'][training_step_id] = mean_log_prob_density
            episode_metrics['value_loss'][training_step_id] = value_loss

            if config.verbosity > 0:
                if training_step_id % 50 == 0:
                    progress_print()

        # log episode results
        episode_mean_sum_rate = mean(episode_metrics['sum_rate_per_step'])
        metrics['mean_sum_rate_per_episode'][training_episode_id] = episode_mean_sum_rate
        if config.verbosity == 1:
            print(f' Episode mean reward: {episode_mean_sum_rate:.4f}'
                  f' std {std(episode_metrics["sum_rate_per_step"]):.2f},'
                  f' current exploration: {mean(episode_metrics["mean_log_prob_density"]):.2f},'
                  f' value loss: {mean(episode_metrics["value_loss"]):.5f}'
                  )

        # save network snapshot
        if training_episode_id > 10 and episode_mean_sum_rate > high_score:
            high_score = mean(episode_metrics['sum_rate_per_step'])
            save_model_checkpoint(extra=episode_mean_sum_rate)

    # end compute performance profiling
    if config.profile:
        profiler.disable()
        if config.verbosity == 1:
            profiler.print_stats(sort='cumulative')
        profiler.dump_stats(Path(config.performance_profile_path, f'{config.config_learner.training_name}.profile'))

    save_model_checkpoint(extra=episode_mean_sum_rate)
    save_results()

    # TODO: Move this to proper place
    plot_sweep(range(config.config_learner.training_episodes), metrics['mean_sum_rate_per_episode'],
               'Training Episode', 'Sum Rate')
    if config.show_plots:
        plt.show()


if __name__ == '__main__':
    cfg = Config()
    train_sac_single_error(config=cfg)
