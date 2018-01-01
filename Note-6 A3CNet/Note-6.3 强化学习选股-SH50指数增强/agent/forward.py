import tensorflow as tf
import sonnet as snt
# from sonnet.python.modules.base import AbstractModule
# from sonnet.python.modules.basic import Linear as snt.Linear
# from sonnet.python.modules.gated_rnn import LSTM as snt.LSTM
# from sonnet.python.modules.basic_rnn import DeepRNN as snt.DeepRNN
# from sonnet.python.modules.basic import BatchApply as snt.BatchApply


def swich(inputs):
    return inputs * tf.nn.sigmoid(inputs)


def Linear(name, output_size):
    initializers = {"w": tf.truncated_normal_initializer(stddev=0.1),
                    "b": tf.constant_initializer(value=0.1)}
    regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=0.1),
                    "b": tf.contrib.layers.l2_regularizer(scale=0.1)}
    return snt.Linear(output_size,
                      initializers=initializers,
                      regularizers=regularizers,
                      name=name)


# def build_common_network(inputs):
#     """common network
#     :param inputs: [Time, Batch, state_size]
#     :return: [Time, Batch, hidden_size]
#     """
#     # build rnn
#     batch_size = inputs.get_shape().as_list()[1]
#     l1 = snt..LSTM(128, name='rnn_first')
#     l2 = snt..LSTM(64, name='rnn_second')
#     l3 = snt..LSTM(32, name='rnn_third')
#     rnn = snt..DeepRNN([l1, l2, l3])
#     initial_state = rnn.initial_state(batch_size)
#     # looping
#     output_sequence, final_state = tf.nn.dynamic_rnn(
#         rnn, inputs, initial_state=initial_state, time_major=True)
#     return output_sequence


def build_common_network(inputs):
    """common network
    :param inputs: [Time, Batch, state_size]
    :return: [Time, Batch, hidden_size]
    """
    # build rnn
    batch_size = inputs.get_shape().as_list()[1]
    l1 = snt.LSTM(64, name='rnn_first')
    l2 = snt.LSTM(32, name='rnn_second')
    rnn = snt.DeepRNN([l1, l2])
    initial_state = rnn.initial_state(batch_size)
    # looping
    output_sequence, final_state = tf.nn.dynamic_rnn(
        rnn, inputs, initial_state=initial_state, time_major=True)
    return output_sequence


class ActorNet(snt.AbstractModule):
    """actor network
    """
    def __init__(self, name='Actor'):
        super().__init__(name=name)

    def _build(self, output_size, inputs):
        # loop net -> [Time, Batch, hidden_size]
        net = build_common_network(inputs)  # rnn output (-1, 1)
        # linear net
        net = snt.BatchApply(Linear('input_layer', 64))(net)
        net = swich(net)
        net = snt.BatchApply(Linear('output_layer', output_size))(net)
        return tf.nn.softmax(net)  # [Time, Batch, output_size]

    def get_regularization(self):
        return self.get_variables(tf.GraphKeys.REGULARIZATION_LOSSES)


class CriticNet(snt.AbstractModule):
    """critic network
    """
    def __init__(self, name='critic'):
        super().__init__(name=name)

    def _build(self, inputs):
        # loop net -> [Time, Batch, hidden_size]
        net = build_common_network(inputs)  # range (-1, 1)
        # linear net
        net = snt.BatchApply(Linear('input_layer', 64))(net)
        net = swich(net)
        net = snt.BatchApply(Linear('output_layer', 1))(net)
        net = tf.squeeze(net, axis=2)
        # net = tf.nn.tanh(net)
        return tf.reduce_mean(net, axis=1)  # [Time]

    def get_regularization(self):
        return self.get_variables(tf.GraphKeys.REGULARIZATION_LOSSES)