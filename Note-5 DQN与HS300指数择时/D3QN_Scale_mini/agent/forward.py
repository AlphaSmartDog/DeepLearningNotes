import tensorflow as tf
import sonnet as snt
from params import *


def Swich(inputs):
    return inputs * tf.nn.sigmoid(inputs)


class Forward(snt.AbstractModule):
    def __init__(self, name='forward'):
        super().__init__(name=name)
        self.name = name

    def _build(self, inputs):
        with tf.variable_scope(self.name):
            net = self._build_shared_network(inputs)
            V = snt.Linear(1, 'value')(net)
            A = snt.Linear(ACTION_SIZE, 'advantage')(net)
            Q = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))
            return Q

    def _build_shared_network(self, inputs):
        # net = snt.Conv2D(32, [8, 8], [4, 4])(inputs)
        # net = Swich(net)
        net = snt.Conv2D(16, [4, 4], [4, 4])(inputs)
        net = Swich(net)
        net = tf.nn.dropout(net, DROPOUT)
        net = snt.Conv2D(32, [3, 3], [3, 3])(net)
        net = Swich(net)
        net = tf.nn.dropout(net, DROPOUT)
        net = snt.BatchFlatten(1)(net)
        net = snt.Linear(64)(net)
        return Swich(net)



