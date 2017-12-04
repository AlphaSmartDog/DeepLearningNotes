import tensorflow as tf
from sonnet.python.modules.basic import Linear as sntLinear

def swich(input):
    return input * tf.nn.sigmoid(input)

def Linear(output_size, name):
    initializers = {"w": tf.truncated_normal_initializer(stddev=0.1),
                    "b": tf.constant_initializer(value=0.1)}
    regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=0.1),
                    "b": tf.contrib.layers.l2_regularizer(scale=0.1)}
    return sntLinear(output_size,
                  initializers=initializers,
                  regularizers=regularizers,
                  name=name)

def get_regularization(scope_name):
    return tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope_name)

def get_trainables(scope_name):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)