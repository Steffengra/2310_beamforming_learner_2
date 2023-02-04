
from logging import (
    Logger,
)
from numpy import (
    ndarray,
    array,
    newaxis,
    concatenate,
)
from numpy.random import (
    Generator,
)
from tensorflow import (
    function as tf_function,
    GradientTape as tf_GradientTape,
    float32 as tf_float32,
    clip_by_value as tf_clip_by_value,
    concat as tf_concat,
    squeeze as tf_squeeze,
    minimum as tf_minimum,
    reduce_mean as tf_reduce_mean,
)
from tensorflow.random import (
    normal as tf_normal,
)

from src.models.network_models import (
    ValueNetwork,
    PolicyNetwork,
)
from src.models.experience_buffer import (
    ExperienceBuffer,
)


class TD3ActorCritic:
    """
    TD3 proposes a few changes to DDPG with the goal of reducing variance and increasing stability

    - "Soft" update to the target network theta_target = tau * theta_primary + (1 - tau) * theta_target_old
    - Update Value a few times after Policy update to have a good Value estimate for the next Policy update
    - Adding small noise to the TD error bootstrap to smoothen the Value estimate
    - Two independently trained critics, clipping the value estimate to
        the smaller of the two to avoid overestimation
    """

    def __init__(
            self,
            rng: Generator,
            parent_logger: Logger,
            training_batch_size: int,
            training_target_update_momentum_tau: float,
            training_minimum_experiences: int,
            training_noise_std: float,
            training_noise_clip: float,
            future_reward_discount_gamma: float,
            experience_buffer_args: dict,
            network_args: dict,
    ) -> None:

        self.rng = rng
        self.logger = parent_logger.getChild(__name__)

        self.training_batch_size: int = training_batch_size
        self.training_target_update_momentum_tau: float = training_target_update_momentum_tau
        self.training_minimum_experiences: int = training_minimum_experiences
        self.training_noise_std: float = training_noise_std
        self.training_noise_clip: float = training_noise_clip
        self.future_reward_discount_gamma: float = future_reward_discount_gamma

        self.experience_buffer = ExperienceBuffer(rng=rng, **experience_buffer_args)

        self.networks: dict = {
            'value': [],
            'policy': [],
        }
        self._initialize_networks(**network_args)

        self.logger.info('TD3 setup complete')

    def _initialize_networks(
            self,
            value_network_args: dict,
            policy_network_args: dict,
            value_network_optimizer,
            value_network_optimizer_args: dict,
            value_network_loss,
            policy_network_optimizer,
            policy_network_optimizer_args: dict,
            policy_network_loss,
            size_state,
            num_actions,
    ) -> None:

        # create networks
        # TODO: The number of value networks should probably be part of config
        #  although tf methods make it annoying to do more than 2

        for _ in range(2):
            self.networks['value'].append(
                {
                    'primary': ValueNetwork(**value_network_args),
                    'target': ValueNetwork(**value_network_args),
                }
            )

        for _ in range(1):
            self.networks['policy'].append(
                {
                    'primary': PolicyNetwork(**policy_network_args),
                    'target': PolicyNetwork(**policy_network_args),
                }
            )

        # compile networks
        dummy_state = self.rng.random(size_state)
        dummy_action = self.rng.random(num_actions)
        for network_type, network_list in self.networks.items():
            # create dummy input
            if network_type == 'policy':
                dummy_input = dummy_state[newaxis]
                optimizer = policy_network_optimizer
                optimizer_args = policy_network_optimizer_args
                loss = policy_network_loss
            elif network_type == 'value':
                dummy_input = concatenate([dummy_state, dummy_action])[newaxis]
                optimizer = value_network_optimizer
                optimizer_args = value_network_optimizer_args
                loss = value_network_loss
            # feed dummy input, compile primary
            for network_pair in network_list:
                for network_rank, network in network_pair.items():
                    network.initialize_inputs(dummy_input)
                network_pair['primary'].compile(
                    optimizer=optimizer(**optimizer_args),
                    # loss=loss,  # TODO: doesnt work anymore when saving/loading currently, probably fixable
                )
        self.update_target_networks(tau_target_update_momentum=1.0)

    @tf_function
    def update_target_networks(
            self,
            tau_target_update_momentum: float,
    ) -> None:

        for network_list in self.networks.values():
            for network_pair in network_list:
                for v_primary, v_target in zip(network_pair['primary'].trainable_variables,
                                               network_pair['target'].trainable_variables):
                    v_target.assign(tau_target_update_momentum * v_primary + (1 - tau_target_update_momentum) * v_target)

    def get_action(
            self,
            state: ndarray,
    ) -> ndarray:

        if state.ndim == 1:
            state = state[newaxis]
        action = self.networks['policy'][0]['primary'].call(state)
        return action.numpy().flatten()

    def add_experience(
            self,
            experience: dict,
    ) -> None:
        self.experience_buffer.add_experience(experience=experience)

    def train(
            self,
            train_policy: bool = True,
    ) -> None:

        if (
            (self.experience_buffer.get_len() < self.training_minimum_experiences)
            or
            (self.experience_buffer.get_len() < self.training_batch_size)
        ):
            return

        # sample from buffer
        (
            sample_experiences,
            sample_experience_ids,
            sample_importance_weights,
        ) = self.experience_buffer.sample(batch_size=self.training_batch_size)

        states = array([experience['state'] for experience in sample_experiences], dtype='float32')
        actions = array([experience['action'] for experience in sample_experiences], dtype='float32')
        rewards = array([experience['reward'] for experience in sample_experiences], dtype='float32')
        next_states = array([experience['next_state'] for experience in sample_experiences], dtype='float32')
        train_policy = array(train_policy)

        self.train_graph(
            train_policy=train_policy,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            sample_importance_weights=sample_importance_weights,
        )

    @tf_function
    def train_graph(
            self,
            train_policy: bool,
            states,
            actions,
            rewards,
            next_states,
            sample_importance_weights,
    ) -> None:
        """
        Wraps as much as possible of the training process into a tf.function graph for performance
        """
        # TRAIN VALUE NETWORKS
        target_q = rewards

        # future reward estimate
        if self.future_reward_discount_gamma > 0:  # future rewards estimate
            next_actions = self.networks['policy'][0]['target'].call(next_states)

            # Add a small amount of random noise to action for smoothing
            noise = tf_normal(shape=next_actions.shape,
                              mean=0, stddev=self.training_noise_std, dtype=tf_float32)
            noise = tf_clip_by_value(noise, -self.training_noise_clip, self.training_noise_clip)
            next_actions += noise

            input_vector = tf_concat([next_states, next_actions], axis=1)

            q_estimate_1 = self.networks['value'][0]['target'].call(input_vector)
            q_estimate_2 = self.networks['value'][1]['target'].call(input_vector)
            conservative_q_estimate = tf_squeeze(tf_minimum(q_estimate_1, q_estimate_2))
            target_q = target_q + self.future_reward_discount_gamma * conservative_q_estimate

        input_vector = tf_concat([states, actions], axis=1)

        if self.future_reward_discount_gamma > 0:  # value net 2 is only required when gamma > 0
            range_value_net_training = 2
        else:
            range_value_net_training = 1

        # gradient steps:
        for network_id in range(range_value_net_training):
            with tf_GradientTape() as tape:  # autograd
                estimate = tf_squeeze(self.networks['value'][network_id]['primary'].call(input_vector))
                td_error = target_q - estimate
                weighted_loss = tf_reduce_mean(sample_importance_weights * td_error**2)

                loss = (
                    weighted_loss
                )
            gradients = tape.gradient(target=loss,  # d_loss / d_parameters
                                      sources=self.networks['value'][network_id]['primary'].trainable_variables)
            self.networks['value'][network_id]['primary'].optimizer.apply_gradients(  # apply gradient update
                zip(gradients, self.networks['value'][network_id]['primary'].trainable_variables))

        # TRAIN POLICY NETWORK
        if train_policy:
            input_vector = states
            with tf_GradientTape() as tape:  # autograd
                # loss value network
                actor_actions = self.networks['policy'][0]['primary'].call(input_vector)
                value_network_input = tf_concat([input_vector, actor_actions], axis=1)
                # Original Paper, DDPG Paper and other implementations train on primary network. Why?
                #  Because otherwise the value net is always one gradient step behind
                value_network_score_1 = tf_reduce_mean(
                    self.networks['value'][0]['primary'].call(value_network_input))
                loss = (
                    - value_network_score_1
                )
            gradients = tape.gradient(target=loss,  # d_loss / d_parameters
                                      sources=self.networks['policy'][0]['primary'].trainable_variables)
            self.networks['policy'][0]['primary'].optimizer.apply_gradients(
                zip(gradients, self.networks['policy'][0]['primary'].trainable_variables))  # apply gradient update

        # UPDATE TARGET NETWORKS
        self.update_target_networks(tau_target_update_momentum=self.training_target_update_momentum_tau)
