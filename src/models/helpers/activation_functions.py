
"""Custom activation functions for neural networks."""

import tensorflow as tf


def activation_penalized_tanh(
        inputs: tf.Tensor
) -> tf.Tensor:

    """
    Some people report that the penalized tanh [1]:
    y = { tanh(x)     | x >= 0
        { a * tanh(x) | x < 0
    works quite well empirically both in consistency and peak performance across
    multiple tasks.

    [1] Revise Saturated Activation Functions, Bing Xu, Ruitong Huang, Mu Li, 2016
    """

    penalty = 0.25
    output = tf.tanh(inputs)
    output = tf.where(tf.less(inputs, 0), tf.multiply(output, penalty), output)

    return output


def activation_shaped_tanh(
        inputs: tf.Tensor
) -> tf.Tensor:

    """
    LeCun recommends in [1] to reshape the standard tanh such that
    1) When transforming the inputs of this tanh to have zero mean and var=1,
        the output of this tanh will also have zero mean and var=1 on average
    2) f(+-1) = +-1
    3) x=1 is a 2nd derivative maximum
    4) Effective gain close to 1

    [1] LeCun, Yann, Leon Bottou, Genevieve B. Orr, and Klaus-Robert Müller. “Efficient BackProp.”, 1998
    """

    a = 1.7159
    b = 2/3

    y = a * tf.tanh(b * inputs)

    return y
