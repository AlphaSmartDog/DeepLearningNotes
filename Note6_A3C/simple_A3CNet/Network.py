import tensorflow as tf
from sonnet.python.modules.base import AbstractModule
from util import Linear, swich


class Network(AbstractModule):
    def __init__(self, name):
        super().__init__(name=name)

    def _build(self, inputs, output_size):
        network = Linear(32, 'input_layer')(inputs)
        network = swich(network)
        network = Linear(64, 'hidden_layer')(network)
        network = swich(network)
        return Linear(output_size, 'output_layer')(network)

    def get_regularization(self):
        return self.get_variables(tf.GraphKeys.REGULARIZATION_LOSSES)
