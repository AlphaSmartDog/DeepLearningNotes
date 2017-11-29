import tensorflow as tf
from sonnet.python.modules.base import AbstractModule
from util import Linear, Conv2D, swich


class FCNet(AbstractModule):
    def __init__(self, name):
        super().__init__(name=name)

    def _build(self, inputs, output_size):
        network = Linear(32, 'input_layer')(inputs)
        network = tf.nn.tanh(network)
        network = Linear(64, 'hidden_layer')(network)
        network = swich(network)
        return Linear(output_size, 'output_layer')(network)

    def get_regularization(self):
        return self.get_variables(tf.GraphKeys.REGULARIZATION_LOSSES)


class ConvNet(AbstractModule):
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
