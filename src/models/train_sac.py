
from datetime import datetime
from pathlib import Path
from sys import path as sys_path
from shutil import (
    copytree,
    rmtree,
)
import gzip
import pickle

import numpy as np
from matplotlib.pyplot import show as plt_show

project_root_path = Path(Path(__file__).parent, '..', '..')
sys_path.append(str(project_root_path.resolve()))

import src
from src.config.config import (
    Config,
)
from src.data.satellite_manager import (
    SatelliteManager,
)
from src.data.user_manager import (
    UserManager,
)
from src.models.algorithms.soft_actor_critic import (
    SoftActorCritic,
)
from src.models.helpers.get_state_norm_factors import (
    get_state_norm_factors,
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
    rad_and_phase_to_complex_vector,
)
from src.utils.norm_precoder import (
    norm_precoder,
)
from src.utils.plot_sweep import (
    plot_sweep,
)
from src.utils.profiling import (
    start_profiling,
    end_profiling,
)
from src.utils.progress_printer import (
    progress_printer,
)
from src.utils.update_sim import (
    update_sim,
)


def train_sac_single_error(
        config: 'src.config.config.Config',
) -> Path:
    """Train a Soft Actor Critic precoder according to the config."""

    def progress_print(to_log: bool = False) -> None:
        progress = (
                (training_episode_id * config.config_learner.training_steps_per_episode + training_step_id + 1)
                / (config.config_learner.training_episodes * config.config_learner.training_steps_per_episode)
        )
        if not to_log:
            progress_printer(progress=progress, real_time_start=real_time_start)
        else:
            progress_printer(progress=progress, real_time_start=real_time_start, logger=logger)

    def policy_training_criterion() -> bool:
        """Train policy networks only every k steps and/or only after j total steps to ensure a good value function"""
        if (
            simulation_step > config.config_learner.train_policy_after_j_steps
            and
            (simulation_step % config.config_learner.train_policy_every_k_steps) == 0
        ):
            return True
        return False

    def value_training_criterion() -> bool:
        """Train value networks only every k steps and/or only after j total steps"""
        if (
            simulation_step > config.config_learner.train_value_after_j_steps
            and
            (simulation_step % config.config_learner.train_value_every_k_steps) == 0
        ):
            return True
        return False

    def add_mmse_experience():

        # this needs to use erroneous csi, otherwise the data distribution in buffer
        #  is changed significantly from reality, i.e., the learner gets too much confidence that
        #  the csi is reliable
        w_mmse = mmse_precoder_normalized(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            **config.mmse_args
        )
        reward_mmse = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_mmse,
            noise_power_watt=config.noise_power_watt,
        )
        if (reward_mmse > reward) or not config.config_learner.only_add_mmse_samples_with_greater_reward:
            mmse_experience = {
                'state': state_current,
                'action': complex_vector_to_double_real_vector(w_mmse.flatten()),
                'reward': reward_mmse,
                'next_state': state_next,
            }
            sac.add_experience(mmse_experience)

    def save_model_checkpoint(extra=None):

        name = f'userwiggle_{config.user_dist_bound}'
        if extra is not None:
            name += f'_snap_{extra:.3f}'
        checkpoint_path = Path(
            config.trained_models_path,
            config.config_learner.training_name,
            'single_error',
            name,
        )

        logger.info(f'Saved model checkpoint at mean reward {extra:.3f}')

        sac.networks['policy'][0]['primary'].save(Path(checkpoint_path, 'model'))

        # save config
        copytree(Path(config.project_root_path, 'src', 'config'),
                 Path(checkpoint_path, 'config'),
                 dirs_exist_ok=True)

        # save norm dict
        with gzip.open(Path(checkpoint_path, 'config', 'norm_dict.gzip'), 'wb') as file:
            pickle.dump(norm_dict, file)

        # clean model checkpoints
        for high_score_prior_id, high_score_prior in enumerate(reversed(high_scores)):
            if high_score > 1.05 * high_score_prior or high_score_prior_id > 3:

                name = f'userwiggle_{config.user_dist_bound}_snap_{high_score_prior:.3f}'

                prior_checkpoint_path = Path(
                    config.trained_models_path,
                    config.config_learner.training_name,
                    'single_error',
                    name
                )
                rmtree(path=prior_checkpoint_path, ignore_errors=True)
                high_scores.remove(high_score_prior)

        return checkpoint_path

    def save_results():

        name = f'training_error_userwiggle_{config.user_dist_bound}.gzip'

        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'single_error')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(results_path, name), 'wb') as file:
            pickle.dump(metrics, file=file)

    logger = config.logger.getChild(__name__)

    satellite_manager = SatelliteManager(config=config)
    user_manager = UserManager(config=config)
    sac = SoftActorCritic(rng=config.rng, **config.config_learner.algorithm_args)

    norm_dict = get_state_norm_factors(config=config, satellite_manager=satellite_manager, user_manager=user_manager)
    logger.info('State normalization factors found')
    logger.info(norm_dict)

    metrics: dict = {
        'mean_sum_rate_per_episode': -np.infty * np.ones(config.config_learner.training_episodes)
    }
    high_score = -np.infty
    high_scores = []

    real_time_start = datetime.now()

    profiler = None
    if config.profile:
        profiler = start_profiling()

    step_experience: dict = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}

    for training_episode_id in range(config.config_learner.training_episodes):

        episode_metrics: dict = {
            'sum_rate_per_step': -np.infty * np.ones(config.config_learner.training_steps_per_episode),
            'mean_log_prob_density': np.infty * np.ones(config.config_learner.training_steps_per_episode),
            'value_loss': -np.infty * np.ones(config.config_learner.training_steps_per_episode),
        }

        update_sim(config, satellite_manager, user_manager)  # todo: we update_sim twice in this script, correct?
        state_next = config.config_learner.get_state(
            satellite_manager=satellite_manager,
            norm_factors=norm_dict['norm_factors'],
            **config.config_learner.get_state_args
        )

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
            # w_precoder_vector = rad_and_phase_to_complex_vector(action)
            w_precoder = w_precoder_vector.reshape((config.sat_nr*config.sat_ant_nr, config.user_nr))
            w_precoder_normed = norm_precoder(precoding_matrix=w_precoder, power_constraint_watt=config.power_constraint_watt,
                                              per_satellite=True, sat_nr=config.sat_nr, sat_ant_nr=config.sat_ant_nr)

            # step simulation based on action, determine reward
            reward = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_precoder_normed,
                noise_power_watt=config.noise_power_watt,
            )
            step_experience['reward'] = reward

            # optionally add the corresponding mmse precoder to the data set
            if config.rng.random() < config.config_learner.percentage_mmse_samples_added_to_exp_buffer:
                add_mmse_experience()  # todo note: currently state_next saved in the mmse experience is not correct

            # update simulation state
            update_sim(config, satellite_manager, user_manager)

            # get new state
            state_next = config.config_learner.get_state(
                satellite_manager=satellite_manager,
                norm_factors=norm_dict['norm_factors'],
                **config.config_learner.get_state_args
            )
            step_experience['next_state'] = state_next

            sac.add_experience(experience=step_experience)

            # train allocator off-policy
            train_policy = False
            if policy_training_criterion():
                train_policy = True
            train_value = False
            if value_training_criterion():
                train_value = True

            if train_value or train_policy:
                mean_log_prob_density, value_loss = sac.train(
                    toggle_train_value_networks=train_value,
                    toggle_train_policy_network=train_policy,
                    toggle_train_entropy_scale_alpha=True,
                )
            else:
                mean_log_prob_density = np.nan
                value_loss = np.nan

            # log results
            episode_metrics['sum_rate_per_step'][training_step_id] = reward
            episode_metrics['mean_log_prob_density'][training_step_id] = mean_log_prob_density
            episode_metrics['value_loss'][training_step_id] = value_loss

            if config.verbosity > 0:
                if training_step_id % 50 == 0:
                    progress_print()

        # log episode results
        episode_mean_sum_rate = np.nanmean(episode_metrics['sum_rate_per_step'])
        metrics['mean_sum_rate_per_episode'][training_episode_id] = episode_mean_sum_rate

        if config.verbosity > 0:
            print('\r', end='')  # clear console for logging results
        progress_print(to_log=True)
        logger.info(
            f'Episode {training_episode_id}:'
            f' Episode mean reward: {episode_mean_sum_rate:.4f}'
            f' std {np.nanstd(episode_metrics["sum_rate_per_step"]):.2f},'
            f' current exploration: {np.nanmean(episode_metrics["mean_log_prob_density"]):.2f},'
            f' value loss: {np.nanmean(episode_metrics["value_loss"]):.5f}'
            f' curr. lr: {sac.networks["policy"][0]["primary"].optimizer.learning_rate(sac.networks["policy"][0]["primary"].optimizer.iterations):.2E}'
        )

        # save network snapshot
        if episode_mean_sum_rate > high_score:
            high_score = episode_mean_sum_rate.copy()
            high_scores.append(high_score)
            best_model_path = save_model_checkpoint(extra=episode_mean_sum_rate)

    # end compute performance profiling
    if profiler is not None:
        end_profiling(profiler)

    save_results()

    # TODO: Move this to proper place
    plot_sweep(range(config.config_learner.training_episodes), metrics['mean_sum_rate_per_episode'],
               'Training Episode', 'Sum Rate')
    if config.show_plots:
        plt_show()

    return best_model_path


if __name__ == '__main__':
    cfg = Config()
    train_sac_single_error(config=cfg)
