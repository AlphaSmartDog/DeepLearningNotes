import tensorflow as tf
import sonnet as snt


def swich(input):
    return input * tf.nn.sigmoid(input)


def Linear(name, output_size):
    initializers = {"w": tf.truncated_normal_initializer(stddev=0.1),
                    "b": tf.constant_initializer(value=0.1)}
    regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=0.1),
                    "b": tf.contrib.layers.l2_regularizer(scale=0.1)}
    return snt.Linear(output_size,
                      initializers=initializers,
                      regularizers=regularizers,
                      name=name)


def Conv2D(name, output_channels, kernel_shape, stride):
    initializers = {"w": tf.truncated_normal_initializer(stddev=0.1),
                    "b": tf.constant_initializer(value=0.1)}
    regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=0.1),
                    "b": tf.contrib.layers.l2_regularizer(scale=0.1)}
    return snt.Conv2D(output_channels,
                      kernel_shape,
                      stride,
                      initializers=initializers,
                      regularizers=regularizers,
                      name=name)


class Forward(snt.AbstractModule):
    def __init__(self, name):
        super().__init__(name=name)

    def _build(self, inputs, output_size):
        network = Conv2D('input_layer', 16, [8, 8], [4, 4])(inputs)
        network = swich(network)
        network = Conv2D('hidden_layer', 32, [4, 4], [2, 2])(network)
        network = swich(network)
        network = tf.contrib.layers.flatten(network)
        network = Linear('final_layer', 64)(network)
        network = swich(network)
        return Linear('output_layer', output_size)(network)

    def get_regularization(self):
        return self.get_variables(tf.GraphKeys.REGULARIZATION_LOSSES)
