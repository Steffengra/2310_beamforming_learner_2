
import tensorflow as tf

from src.models.helpers.get_state import (
    get_state_erroneous_channel_state_information,
    get_state_aods,
)


class ConfigSACLearner:
    """Defines parameters to use when learning with Soft Actor Critic."""

    def __init__(
            self,
            sat_nr,
            sat_ant_nr,
            user_nr,
    ) -> None:

        self.training_name: str = (
            '1_sat'
            '_10_ant'
            '_3_usr'
            '_10000_dist'
            '_0.05_error_on_cos'
            '_0.1_fading'
        )

        self.get_state = get_state_erroneous_channel_state_information
        self.get_state_args = {
            'csi_format': 'rad_phase',  # 'rad_phase', 'rad_phase_reduced', 'real_imag'
            'norm_state': True,  # !!HEURISTIC!!, this will break if you dramatically change the setup
        }
        self.get_state_norm_factors_iterations: int = 100_000  # how many samples to calculate means and stds

        self.percentage_mmse_samples_added_to_exp_buffer: float = 0.0  # [0.0, 1.0] chance for mmse action to be added
        self.only_add_mmse_samples_with_greater_reward: bool = True  # only add samples with reward_mmse > reward_sac

        self.training_args: dict = {
            'future_reward_discount_gamma': 0.0,  # Exponential future reward discount for stability
            'entropy_scale_alpha_initial': 1,  # weights the 'soft' entropy penalty against the td error
            'target_entropy': 1.0,  # SAC heuristic impl. = product of action_space.shape
            'entropy_scale_optimizer': tf.keras.optimizers.SGD,
            'entropy_scale_optimizer_args': {
                'learning_rate': 1e-3,  # LR=0.0 -> No adaptive entropy scale -> manually tune initial entropy scale
            },
            'training_minimum_experiences': 1_000,
            'training_batch_size': 1024,
            'training_target_update_momentum_tau': 0,  # How much of the primary network copy to target networks
            'training_l2_norm_scale_value': 0.01,
            'training_l2_norm_scale_policy': 0.01,  # related to weight decay
        }
        self.experience_buffer_args: dict = {
            'buffer_size': 100_000,
            'priority_scale_alpha': 0.0,  # alpha in [0, 1], alpha=0 uniform sampling, 1 is fully prioritized sampling
            'importance_sampling_correction_beta': 1.0  # beta in [0%, 100%], beta=100% is full correction
        }
        self.train_policy_every_k_steps: int = 10  # train policy only every k steps
        self.train_policy_after_j_steps: int = 0  # start training policy after j steps
        self.train_value_every_k_steps: int = 10  # train value only every k steps
        self.train_value_after_j_steps: int = 0  # start training value after j steps
        self.network_args: dict = {
            'value_network_args': {
                'hidden_layer_units': [512, 512, 512, 512, ],
                'activation_hidden': 'leaky_relu',  # >'relu', 'leaky_relu', 'tanh', 'penalized_tanh', 'shaped_tanh'
                'kernel_initializer_hidden': 'glorot_uniform',  # >'glorot_uniform', 'he_uniform'
                'batch_norm_input': False,  # might not be a good idea due to 32bit input
                'batch_norm': True,  # too high learning rate will lead to problems w/ weights changing too fast
            },
            'value_network_optimizer': tf.keras.optimizers.Adam,
            'value_network_optimizer_args': {
                # 'learning_rate': 1e-3,
                'learning_rate': tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=7e-5,
                                                                                   first_decay_steps=100),
                'amsgrad': False,
            },
            'policy_network_args': {
                'hidden_layer_units': [512, 512, 512, 512, ],
                'activation_hidden': 'penalized_tanh',  # >'relu', 'leaky_relu', 'tanh', 'penalized_tanh', 'shaped_tanh'
                'kernel_initializer_hidden': 'glorot_uniform',  # >'glorot_uniform', 'he_uniform'
                'batch_norm_input': False,  # might not be a good idea due to 32bit input
                'batch_norm': True,  # too high learning rate will lead to problems w/ weights changing too fast
            },
            'policy_network_optimizer': tf.keras.optimizers.Adam,
            'policy_network_optimizer_args': {
                # 'learning_rate': 1e-4,
                'learning_rate': tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=7e-6,
                                                                                   first_decay_steps=100),
                'amsgrad': True,
            },
        }

        # TRAINING
        self.training_episodes: int = 20_000  # a new episode is a full reset of the simulation environment
        self.training_steps_per_episode: int = 1_000

        self._post_init(sat_nr=sat_nr, sat_ant_nr=sat_ant_nr, user_nr=user_nr)

    def _post_init(
            self,
            sat_nr,
            sat_ant_nr,
            user_nr,
    ) -> None:

        self.training_args['training_minimum_experiences'] = max(self.training_args['training_minimum_experiences'],
                                                                 self.training_args['training_batch_size'])

        self.update(sat_nr=sat_nr, sat_ant_nr=sat_ant_nr, user_nr=user_nr)

        # Collected args
        self.algorithm_args = {
            **self.training_args,
            'network_args': self.network_args,
            'experience_buffer_args': self.experience_buffer_args,
        }

    def update(
            self,
            sat_nr: int,
            sat_ant_nr: int,
            user_nr: int,
    ) -> None:
        """Update those config parameters that are calculated from others."""

        if self.get_state == get_state_aods:
            self.network_args['size_state'] = sat_nr * user_nr
        elif self.get_state == get_state_erroneous_channel_state_information:
            if self.get_state_args['csi_format'] in ['rad_phase', 'real_imag']:
                self.network_args['size_state'] = 2 * sat_nr * sat_ant_nr * user_nr
            elif self.get_state_args['csi_format'] == 'rad_phase_reduced':
                self.network_args['size_state'] = sat_nr * sat_ant_nr * user_nr + sat_nr * user_nr
        self.network_args['num_actions'] = 2 * sat_nr * sat_ant_nr * user_nr
