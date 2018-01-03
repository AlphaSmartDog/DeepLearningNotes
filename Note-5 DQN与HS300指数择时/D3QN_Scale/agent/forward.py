import tensorflow as tf
import sonnet as snt
from config import *


def swich(inputs):
    return inputs * tf.nn.sigmoid(inputs)


class Forward(snt.AbstractModule):
    def __init__(self, name='forward'):
        super().__init__(name=name)
        self.name = name

    def _build(self, inputs, dropout):
        with tf.variable_scope(self.name):
            net = self._build_shared_network(inputs, dropout)
            V = snt.Linear(1, 'value')(net)
            A = snt.Linear(ACTION_SIZE, 'advantage')(net)
            Q = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))
            return Q

    def _build_shared_network(self, inputs, dropout):
        net = snt.Conv2D(32, [8, 8], [4, 4])(inputs)
        net = tf.nn.relu(net)
        net = tf.nn.dropout(net, dropout)

        net = snt.Conv2D(64, [4, 4], [2, 2])(net)
        net = tf.nn.relu(net)
        net = tf.nn.dropout(net, dropout)

        net = snt.Conv2D(64, [3, 3], [1, 1])(net)
        net = tf.nn.relu(net)
        net = tf.nn.dropout(net, dropout)

        net = snt.BatchFlatten(1)(net)
        net = snt.Linear(512)(net)
        return tf.nn.relu(net)



