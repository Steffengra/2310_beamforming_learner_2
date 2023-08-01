
import tensorflow as tf


def activation_penalized_tanh(
        x: tf.Tensor
) -> tf.Tensor:
    y = tf.tanh(x)
    y = tf.where(tf.less(x, 0), tf.multiply(y, .25), y)

    return y
