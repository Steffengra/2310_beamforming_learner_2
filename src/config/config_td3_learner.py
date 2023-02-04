
from numpy import (
    ceil,
)

from src.models.dl_internals_with_expl import (
    optimizer_adam,
    optimizer_nadam,
    mse_loss,
    huber_loss,
)


class ConfigTD3Learner:
    """
    Defines parameters for the TD3 Learner
    """

    def __init__(
            self,
            size_state: int,
            num_actions: int,
    ) -> None:
        # GENERAL
        self.training_name: str = 'test'

        # NETWORKS
        self.hidden_layers_value_net: list = [256, 256, 256]  # + output layer width 1
        self.hidden_layers_policy_net: list = [128, 128, 128]  # + output layer width num_actions
        self.activation_hidden_layers: str = 'tanh'  # [>'relu', 'elu', 'tanh' 'penalized_tanh']
        self.kernel_initializer_hidden_layers: str = 'glorot_uniform'  # options: tf.keras.initializers, default: >'glorot_uniform'
        self.network_loss = mse_loss  # mse_loss, huber_loss

        self.value_network_optimizer = optimizer_adam  # optimizer_adam, optimizer_nadam
        self.value_network_optimizer_args: dict = {  # these change depending on choice of optimizer
            'learning_rate': 1e-4,
            'epsilon': 1e-8,
            'amsgrad': False,
        }
        self.policy_network_optimizer = optimizer_adam
        self.policy_network_optimizer_args: dict = {
            'learning_rate': 1e-6,
            'epsilon': 1e-8,
            'amsgrad': False,
        }

        # TRAINING
        self.training_episodes: int = 100  # a new episode is a full reset of the simulation environment
        self.training_steps_per_episode: int = 1_000

        self.exploration_noise_momentum_initial: float = 1.0
        self.exploration_noise_decay_start_percent: float = 0.0  # After which % of training to start decay
        self.exploration_noise_decay_threshold_percent: float = 0.8  # after which % of training noise == 0

        self.training_batch_size: int = 256
        self.future_reward_discount_gamma: float = 0.0
        self.training_minimum_experiences: int = 0
        self.training_target_update_momentum_tau: float = 0.1

        self.train_policy_every_k_steps: int = 1  # train policy only every k steps to give value approx. time to settle
        self.train_policy_after_j_steps: int = 0  # start training policy only after value approx. starts being sensible

        self.training_noise_std: float = 0.0  # introduce a small amount of noise onto the future policy in value training..
        self.training_noise_clip: float = 0.05  # ..to avoid narrow peaks in value function

        self.experience_buffer_size: int = 70_000
        self.experience_prioritization_factors: dict = {'alpha': 0.0,  # priority = priority ** alpha, \alpha \in [0, 1]
                                                        'beta': 1.0}  # IS correction, \beta \in [0%, 100%]

        self._post_init(
            num_actions=num_actions,
            size_state=size_state,
        )

    def _post_init(
            self,
            num_actions,
            size_state,
    ) -> None:

        self.num_actions: int = num_actions  # == number of outputs of policy network
        self.size_state: int = size_state

        # Arithmetic
        self.exploration_noise_step_start_decay: int = ceil(
            self.exploration_noise_decay_start_percent * self.training_episodes * self.training_steps_per_episode
        )

        self.exploration_noise_linear_decay_per_step: float = (
            self.exploration_noise_momentum_initial / (
                self.exploration_noise_decay_threshold_percent * (
                    self.training_episodes * self.training_steps_per_episode - self.exploration_noise_step_start_decay
                )
            )
        )

        # Collected args
        self.experience_buffer_args: dict = {
            'buffer_size': self.experience_buffer_size,
            'priority_scale_alpha': self.experience_prioritization_factors['alpha'],
            'importance_sampling_correction_beta': self.experience_prioritization_factors['beta'],
        }

        self.network_args: dict = {
            'value_network_args': {
                'hidden_layer_units': self.hidden_layers_value_net,
                'activation_hidden': self.activation_hidden_layers,
                'kernel_initializer_hidden': self.kernel_initializer_hidden_layers,
            },
            'policy_network_args': {
                'num_actions': self.num_actions,
                'hidden_layer_units': self.hidden_layers_policy_net,
                'activation_hidden': self.activation_hidden_layers,
                'kernel_initializer_hidden': self.kernel_initializer_hidden_layers,
            },
            'size_state': self.size_state,
            'num_actions': self.num_actions,
            'value_network_optimizer': self.value_network_optimizer,
            'value_network_optimizer_args': self.value_network_optimizer_args,
            'value_network_loss': self.network_loss,
            'policy_network_optimizer': self.policy_network_optimizer,
            'policy_network_optimizer_args': self.policy_network_optimizer_args,
            'policy_network_loss': self.network_loss,
        }

        self.td3_args: dict = {
            'training_batch_size': self.training_batch_size,
            'training_target_update_momentum_tau': self.training_target_update_momentum_tau,
            'training_minimum_experiences': self.training_minimum_experiences,
            'training_noise_std': self.training_noise_std,
            'training_noise_clip': self.training_noise_clip,
            'future_reward_discount_gamma': self.future_reward_discount_gamma,
            'experience_buffer_args': self.experience_buffer_args,
            'network_args': self.network_args,
        }
