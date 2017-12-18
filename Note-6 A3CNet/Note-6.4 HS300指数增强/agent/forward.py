import tensorflow as tf
# import sonnet as snt
from sonnet.python.modules.base import AbstractModule as sntAbstractModule
from sonnet.python.modules.basic import BatchFlatten as sntBatchFlatten
from sonnet.python.modules.basic import BatchApply as sntBatchApply
from sonnet.python.modules.basic import Linear as sntLinear
from agent.Conv2DLSTM import StackedRNN as _StackedRNN
from params import *


def _swich(inputs):
    return inputs * tf.nn.sigmoid(inputs)


class ActorCriticNet(sntAbstractModule):
    def __init__(self, name='Actor_Critic'):
        super().__init__(name=name)

    def _build_rnn(self, inputs):
        batch_size = inputs.get_shape().as_list()[1]
        image_shape = inputs.get_shape().as_list()[2:4]

        # RNNCore
        rnn = _StackedRNN(RNN_OUT_SHAPE, image_shape, 'RNNCore')
        init = rnn.initial_state(batch_size, tf.float32)
        output_sequence, final_state = tf.nn.dynamic_rnn(
            rnn, inputs, initial_state=init, time_major=True)
        return output_sequence

    def _build_linear(self, inputs):
        # image summary
        net = tf.transpose(inputs, [0, 1, 4, 2, 3])
        net = sntBatchFlatten(3)(net)
        net = sntBatchApply(sntLinear(1), n_dims=3)(net)
        net = _swich(net)
        net = tf.squeeze(net, axis=3)

        # state summary
        net = sntBatchApply(sntLinear(LINEAR_SHAPE), n_dims=2)(net)
        return _swich(net)

    def _build(self, inputs, output_size):
        # rnn
        net = self._build_rnn(inputs)
        net = self._build_linear(net)
        # interface for actor and critic
        actor = sntBatchApply(sntLinear(output_size))(net)
        actor = tf.nn.softmax(actor)
        critic = sntBatchApply(sntLinear(1))(net)
        critic = tf.reduce_mean(critic, axis=1)
        return actor, critic

    def get_regularizer(self):
        return self.get_variables(tf.GraphKeys.REGULARIZATION_LOSSES)