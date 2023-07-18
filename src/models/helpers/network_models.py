
import tensorflow as tf
import tensorflow_probability as tf_p

from src.models.helpers.activation_functions import (
    activation_penalized_tanh,
)


class ValueNetwork(tf.keras.Model):

    def __init__(
            self,
            hidden_layer_units: list,
            activation_hidden: str,
            kernel_initializer_hidden: str
    ):
        super().__init__()
        # Activation----------------------------------------------------------------------------------------------------
        if activation_hidden == 'penalized_tanh':
            activation_hidden = activation_penalized_tanh
        # --------------------------------------------------------------------------------------------------------------

        # Layers--------------------------------------------------------------------------------------------------------
        self.hidden_layers = []
        for size in hidden_layer_units:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    size,
                    activation=activation_hidden,
                    kernel_initializer=kernel_initializer_hidden,  # default: 'glorot_uniform'
                    bias_initializer='zeros'  # default: 'zeros'
                ))

        self.output_layer = tf.keras.layers.Dense(1, dtype=tf.float32)
        # --------------------------------------------------------------------------------------------------------------

    @tf.function
    def call(
            self,
            inputs
    ) -> tf.Tensor:
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)

        return output

    def initialize_inputs(
            self,
            inputs
    ) -> None:
        """
        Ensure each method is traced once for saving
        """
        self(inputs)
        self.call(inputs)


class PolicyNetwork(tf.keras.Model):

    def __init__(
            self,
            hidden_layer_units: list,
            num_actions: int,
            activation_hidden: str,
            kernel_initializer_hidden: str
    ) -> None:
        super().__init__()
        # Activation----------------------------------------------------------------------------------------------------
        if activation_hidden == 'penalized_tanh':
            activation_hidden = activation_penalized_tanh
        # --------------------------------------------------------------------------------------------------------------

        # Layers--------------------------------------------------------------------------------------------------------
        self.hidden_layers = []
        for size in hidden_layer_units:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    size,
                    activation=activation_hidden,
                    kernel_initializer=kernel_initializer_hidden,  # default: 'glorot_uniform'
                    bias_initializer='zeros'  # default: 'zeros'
                ))

        self.output_layer = tf.keras.layers.Dense(num_actions,
                                                  # activation='softmax',
                                                  dtype=tf.float32)
        # --------------------------------------------------------------------------------------------------------------

    @tf.function
    def call(
            self,
            inputs,
    ) -> tf.Tensor:
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)

        return output

    def initialize_inputs(
            self,
            inputs
    ) -> None:
        """
        Ensure each method is traced once for saving
        """
        self(inputs)
        self.call(inputs)


class PolicyNetworkSoft(tf.keras.Model):
    def __init__(
            self,
            num_actions: int,
            hidden_layer_units: list,
            activation_hidden: str = 'relu',
            kernel_initializer_hidden: str = 'glorot_uniform',
    ) -> None:
        super().__init__()

        if activation_hidden == 'penalized_tanh':
            activation_hidden = activation_penalized_tanh

        self.hidden_layers: list = []
        for units in hidden_layer_units:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    units=units,
                    kernel_initializer=kernel_initializer_hidden,  # default='glorot_uniform'
                    activation=activation_hidden,  # default=None
                    bias_initializer='zeros',  # default='zeros'
                )
            )
        self.output_layer_means = tf.keras.layers.Dense(units=num_actions, dtype=tf.float32)
        self.output_layer_log_stds = tf.keras.layers.Dense(units=num_actions, dtype=tf.float32)

    @tf.function
    def call(
            self,
            inputs,
            training=None,
            masks=None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        means = self.output_layer_means(x)
        log_stds = self.output_layer_log_stds(x)

        # NOTE: log_stds are typically clipped in implementations. [-20, 2] seems to be the popular interval.
        #  Clipping logs by such a wide range should not have much of an impact.
        log_stds = tf.clip_by_value(log_stds, -20, 2)

        return (
            means,
            log_stds
        )

    @tf.function
    def get_action_and_log_prob_density(
            self,
            state,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if state.shape.ndims == 1:
            state = tf.expand_dims(state, axis=0)

        means, log_stds = self.call(state)
        stds = tf.exp(log_stds)
        distributions = tf_p.distributions.Normal(loc=means, scale=stds)
        actions = distributions.sample()
        action_log_prob_densities = distributions.log_prob(actions)

        return (
            actions,
            action_log_prob_densities,
        )

    def initialize_inputs(
            self,
            inputs
    ) -> None:
        """
        Ensure each method is traced once for saving
        """
        self(inputs)
        self.call(inputs)
