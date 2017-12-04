import tensorflow as tf
from sonnet.python.modules.basic import Linear as sntLinear
from sonnet.python.modules.conv import Conv2D as sntConv2D


def swich(input):
    return input * tf.nn.sigmoid(input)


def Linear(name, output_size):
    initializers = {"w": tf.truncated_normal_initializer(stddev=0.1),
                    "b": tf.constant_initializer(value=0.1)}
    regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=0.1),
                    "b": tf.contrib.layers.l2_regularizer(scale=0.1)}
    return sntLinear(output_size,
                  initializers=initializers,
                  regularizers=regularizers,
                  name=name)


def Conv2D(name, output_channels, kernel_shape, stride):
    initializers = {"w": tf.truncated_normal_initializer(stddev=0.1),
                    "b": tf.constant_initializer(value=0.1)}
    regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=0.1),
                    "b": tf.contrib.layers.l2_regularizer(scale=0.1)}
    return sntConv2D(output_channels,
                     kernel_shape,
                     stride,
                     initializers=initializers,
                     regularizers=regularizers,
                     name=name)


def get_regularization(scope_name):
    return tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope_name)


def get_trainable(scope_name):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)