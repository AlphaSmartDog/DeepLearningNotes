# fully connected layers
import tensorflow as tf
from sonnet.python.modules.basic import Linear
from sonnet.python.modules.base import AbstractModule

def swich(tensor):
    return tensor * tf.nn.sigmoid(tensor)

def _Linear(output_size, name):
    initializers = {"w": tf.truncated_normal_initializer(stddev=1.0),
                    "b": tf.constant_initializer(value=1.0)}
    regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=0.1),
                    "b": tf.contrib.layers.l2_regularizer(scale=0.1)}
    return Linear(output_size,
                  initializers=initializers,
                  regularizers=regularizers,
                  name=name)

def _build_network(inputs, output_size):
    network = _Linear(32, 'inputs_layer')(inputs)
    network = swich(network)
    network = _Linear(64, 'hidden_layer')(network)
    network = swich(network)
    return _Linear(output_size, 'output_layer')(network)

class FCNet(AbstractModule):
    def __init__(self, name):
        super().__init__(name=name)

    def _build(self, inputs, output_size):
        return _build_network(inputs, output_size)

    def get_regularizers(self):
        collection = tf.GraphKeys.REGULARIZATION_LOSSES
        return self.get_variables(collection)

