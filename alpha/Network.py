import tensorflow as tf
# import sonnet as snt
from sonnet.python.modules.basic import Linear as sntLinear
from sonnet.python.modules.gated_rnn import LSTM as sntLSTM
from sonnet.python.modules.basic_rnn import DeepRNN
from sonnet.python.modules.basic import BatchApply
from sonnet.python.modules.base import AbstractModule


def swich(inputs):
    return inputs * tf.nn.sigmoid(inputs)


def linear(name, output_size):
    initializers = {"w": tf.truncated_normal_initializer(stddev=0.1),
                    "b": tf.constant_initializer(value=0.1)}
    regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=0.1),
                    "b": tf.contrib.layers.l2_regularizer(scale=0.1)}
    return sntLinear(output_size,
                      initializers=initializers,
                      regularizers=regularizers,
                      name=name)


def LSTM(name, hidden_size):
    # initializers = {"w_gates": tf.truncated_normal_initializer(stddev=0.1),
    #                 "b_gates": tf.constant_initializer(value=0.1)}
    # regularizers = {"w_gates": tf.contrib.layers.l2_regularizer(scale=0.1),
    #                 "b_gates": tf.contrib.layers.l2_regularizer(scale=0.1)}
    hidden_clip_value = 50
    cell_clip_value = 50
    return sntLSTM(hidden_size=hidden_size,
                    hidden_clip_value=hidden_clip_value,
                    cell_clip_value=cell_clip_value,
                    name=name)


def build_common_network(name, inputs):
    """common network
    :param inputs: [Time, Batch, state_size]
    :return: [Time, Batch, hidden_size]
    """
    with tf.variable_scope(name):
        # build rnn
        batch_size = inputs.get_shape().as_list()[1]
        l1 = LSTM('rnn_first', 32)
        l2 = LSTM('rnn_second', 64)
        # l3 = LSTM('rnn_third', 128)
        rnn = DeepRNN([l1, l2])
        initial_state = rnn.initial_state(batch_size)
        # looping
        output_sequence, final_state = tf.nn.dynamic_rnn(
            rnn, inputs, initial_state=initial_state, time_major=True)
        return output_sequence


# class SharedNet(snt.AbstractModule):
#
#     def __init__(self, name):
#         super().__init__(name=name)
#
#     def _build(self, inputs):
#         """common network
#         :param inputs: [Time, Batch, state_size]
#         :return: [Time, Batch, hidden_size]
#         """
#         # build rnn
#         batch_size = inputs.get_shape().as_list()[1]
#         l1 = LSTM('rnn_first', 32)
#         l2 = LSTM('rnn_second', 64)
#         # l3 = LSTM('rnn_third', 128)
#         rnn = snt.DeepRNN([l1, l2])
#         initial_state = rnn.initial_state(batch_size)
#         # looping
#         output_sequence, final_state = tf.nn.dynamic_rnn(
#             rnn, inputs, initial_state=initial_state, time_major=True)
#         return output_sequence


class ActorNet(AbstractModule):
    """actor network
    """
    def __init__(self, name='actor'):
        super().__init__(name=name)

    def _build(self, output_size, inputs):
        net = build_common_network('actor', inputs)
        net = BatchApply(linear('input_layer', 128))(net)
        net = swich(net)
        return BatchApply(linear('output_layer', output_size))(net)

    def get_regularization(self):
        return self.get_variables(tf.GraphKeys.REGULARIZATION_LOSSES)


class CriticNet(AbstractModule):
    """critic network
    """
    def __init__(self, name='critic'):
        super().__init__(name=name)

    def _build(self, inputs):
        net = build_common_network('critic', inputs)
        net = BatchApply(linear('input_layer', 128))(net)
        net = tf.tanh(net)
        net = BatchApply(linear('output_layer', 1))(net)
        net = tf.tan(net)
        return tf.reduce_mean(net, axis=1)

    def get_regularization(self):
        return self.get_variables(tf.GraphKeys.REGULARIZATION_LOSSES)