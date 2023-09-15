
import tensorflow as tf
import tensorflow_probability as tf_p

from src.models.helpers.activation_functions import (
    activation_penalized_tanh,
    activation_shaped_tanh,
)


class ValueNetwork(tf.keras.Model):

    def __init__(
            self,
            hidden_layer_units: list,
            activation_hidden: str,
            kernel_initializer_hidden: str,
            batch_norm_input: bool,
            batch_norm: bool,
    ) -> None:
        super().__init__()

        self.batch_norm_input = batch_norm_input
        self.batch_norm = batch_norm

        # Activation----------------------------------------------------------------------------------------------------
        if activation_hidden == 'penalized_tanh':
            activation_hidden = activation_penalized_tanh
        if activation_hidden == 'shaped_tanh':
            activation_hidden = activation_shaped_tanh
        # --------------------------------------------------------------------------------------------------------------

        # Layers--------------------------------------------------------------------------------------------------------
        if self.batch_norm_input:
            self.batch_norm_input_layer = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.hidden_layers = []
        for size in hidden_layer_units:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    size,
                    activation=activation_hidden,
                    kernel_initializer=kernel_initializer_hidden,  # default: 'glorot_uniform'
                    bias_initializer='zeros',  # default: 'zeros'
                )
            )

        if self.batch_norm:
            self.batch_norm_layers = []
            for _ in hidden_layer_units:
                self.batch_norm_layers.append(
                    tf.keras.layers.BatchNormalization(center=False, scale=False)
                )

        self.output_layer = tf.keras.layers.Dense(1, dtype=tf.float32)
        # --------------------------------------------------------------------------------------------------------------

    @tf.function
    def call(
            self,
            inputs,
            training=False,
    ) -> tf.Tensor:

        x = inputs
        if self.batch_norm_input:
            x = self.batch_norm_input_layer(x, training=training)

        if self.batch_norm:
            for layer, norm_layer in zip(self.hidden_layers, self.batch_norm_layers):
                x = layer(x)
                x = norm_layer(x, training=training)
        else:
            for layer in self.hidden_layers:
                x = layer(x)
        output = self.output_layer(x)

        return output

    def initialize_inputs(
            self,
            inputs,
    ) -> None:
        """Ensure each method is traced once for saving."""

        self(inputs)
        self.call(inputs)


class PolicyNetwork(tf.keras.Model):

    def __init__(
            self,
            hidden_layer_units: list,
            num_actions: int,
            batch_norm_input: bool,
            batch_norm: bool,
            activation_hidden: str,
            kernel_initializer_hidden: str,
    ) -> None:
        super().__init__()

        self.batch_norm_input = batch_norm_input
        self.batch_norm = batch_norm

        # Activation----------------------------------------------------------------------------------------------------
        if activation_hidden == 'penalized_tanh':
            activation_hidden = activation_penalized_tanh
        if activation_hidden == 'shaped_tanh':
            activation_hidden = activation_shaped_tanh
        # --------------------------------------------------------------------------------------------------------------

        # Layers--------------------------------------------------------------------------------------------------------
        if self.batch_norm_input:
            self.batch_norm_input_layer = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.hidden_layers = []
        for size in hidden_layer_units:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    size,
                    activation=activation_hidden,
                    kernel_initializer=kernel_initializer_hidden,  # default: 'glorot_uniform'
                    bias_initializer='zeros'  # default: 'zeros'
                ))

        if self.batch_norm:
            self.batch_norm_layers: list = []
            for _ in hidden_layer_units:
                self.batch_norm_layers.append(
                    tf.keras.layers.BatchNormalization(center=False, scale=False)
                )

        self.output_layer = tf.keras.layers.Dense(num_actions,
                                                  # activation='softmax',
                                                  dtype=tf.float32)
        # --------------------------------------------------------------------------------------------------------------

    @tf.function
    def call(
            self,
            inputs,
            training=None,
    ) -> tf.Tensor:

        x = inputs
        if self.batch_norm_input:
            x = self.batch_norm_input_layer(x, training=training)

        if self.batch_norm:
            for layer, norm_layer in zip(self.hidden_layers, self.batch_norm_layers):
                x = layer(x)
                x = norm_layer(x, training=training)
        else:
            for layer in self.hidden_layers:
                x = layer(x)

        output = self.output_layer(x)

        return output

    def initialize_inputs(
            self,
            inputs,
    ) -> None:
        """Ensure each method is traced once for saving."""
        self(inputs)
        self.call(inputs)


class PolicyNetworkSoft(tf.keras.Model):

    """
    Soft network has two output heads per output value.
    One head outputs a mean, the other a (log) standard deviation.
    Larger standard deviations lead to, __on average__, a lower (higher magnitude) log probability.
    Optimizing for higher log probability therefore promotes higher log probabilities.
    """

    def __init__(
            self,
            num_actions: int,
            hidden_layer_units: list,
            batch_norm_input: bool,
            batch_norm: bool,
            activation_hidden: str = 'relu',
            kernel_initializer_hidden: str = 'glorot_uniform',
    ) -> None:
        super().__init__()

        self.batch_norm_input = batch_norm_input
        self.batch_norm = batch_norm

        if activation_hidden == 'penalized_tanh':
            activation_hidden = activation_penalized_tanh
        if activation_hidden == 'shaped_tanh':
            activation_hidden = activation_shaped_tanh

        if self.batch_norm_input:
            self.batch_norm_input_layer = tf.keras.layers.BatchNormalization(center=False, scale=False)

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

        if self.batch_norm:
            self.batch_norm_layers: list = []
            for _ in hidden_layer_units:
                self.batch_norm_layers.append(
                    tf.keras.layers.BatchNormalization(center=False, scale=False)
                )

        self.output_layer_means = tf.keras.layers.Dense(units=num_actions, dtype=tf.float32)
        self.output_layer_log_stds = tf.keras.layers.Dense(units=num_actions, dtype=tf.float32)

    @tf.function
    def call(
            self,
            inputs,
            training=None,
            masks=None,
            print_stds=False,
    ) -> tuple[tf.Tensor, tf.Tensor]:

        x = inputs
        if self.batch_norm_input:
            x = self.batch_norm_input_layer(x, training=training)

        if self.batch_norm:
            for layer, norm_layer in zip(self.hidden_layers, self.batch_norm_layers):
                x = layer(x)
                x = norm_layer(x, training=training)
        else:
            for layer in self.hidden_layers:
                x = layer(x)
        means = self.output_layer_means(x)
        log_stds = self.output_layer_log_stds(x)

        # NOTE: log_stds are typically clipped in implementations. [-20, 2] seems to be the popular interval.
        #  Clipping logs by such a wide range should not have much of an impact.
        log_stds = tf.clip_by_value(log_stds, -20, 2)

        if print_stds:
            tf.print(tf.exp(log_stds))

        return (
            means,
            log_stds,
        )

    @tf.function
    def get_action_and_log_prob_density(
            self,
            state,
            training=False,
            print_stds=False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if state.shape.ndims == 1:
            state = tf.expand_dims(state, axis=0)

        means, log_stds = self.call(state, training=training, print_stds=print_stds)
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
            inputs,
    ) -> None:
        """Ensure each method is traced once for saving."""

        self(inputs)
        self.call(inputs)
