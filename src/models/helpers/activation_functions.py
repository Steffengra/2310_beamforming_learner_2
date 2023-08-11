
import tensorflow as tf


def activation_penalized_tanh(
        x: tf.Tensor
) -> tf.Tensor:

    """
    Some people report that the penalized tanh [1]:
    y = { tanh(x)     | x >= 0
        { a * tanh(x) | x < 0
    works quite well empirically both in consistency and peak performance across
    multiple tasks.

    [1] Revise Saturated Activation Functions, Bing Xu, Ruitong Huang, Mu Li, 2016
    """

    a = 0.25
    y = tf.tanh(x)
    y = tf.where(tf.less(x, 0), tf.multiply(y, a), y)

    return y
