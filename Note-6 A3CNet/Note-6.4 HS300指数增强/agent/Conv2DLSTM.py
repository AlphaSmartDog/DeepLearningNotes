"""Convolutional LSTM
针对Time × 240min x 58fac 量价数据封装类，
并参考GoogLeNet对inputs使用两层1D卷积核降维

猫狗大战 aidabloc@163.com
2017-11-30
"""

import numpy as np
import tensorflow as tf
# import sonnet as snt
from sonnet.python.modules.rnn_core import RNNCore as sntRNNCore
from sonnet.python.modules.conv import Conv2D as sntConv2D
from params import *


def _swich(inputs):
    return inputs * tf.nn.sigmoid(inputs)


class Conv2DLSTM(sntRNNCore):
    """Conv2D-LSTM
    针对处理240分钟线量价因子
    """
    def __init__(self, name,
                 output_channels,
                 input_size):
        """
        :param name: str, name for variable_scope
        :param output_channels: int same as hidden_size
        :param input_size: list, inputs image shape, [240, 58]
        """
        super().__init__(name=name)
        self.output_channels = output_channels
        self.image_shape = self.get_filtered_shape(input_size)

    def _build(self, inputs, prev_state):
        """
        build rnn network
        use tf.nn.dynamic_rnn and make sure time_major=True

        :param inputs: [Batch, Rows, Columns] or [Batch, Height, Width]
        :param prev_state: tuple (prev_hidden, prev_cell)
        prev_hidden and prev_cell get from self.initial_state

        :return: output, state
        output: next_hidden [Batch, new_rows, new_cols, output_channels]
        state = (next_hidden, next_cell)
        next_cell: [Batch, new_rows, new_cols, output_channels]
        """
        # 调整格式
        prev_hidden, prev_cell = prev_state
        # inputs = tf.expand_dims(inputs, axis=-1)

        # 卷积部分
        inputs_and_hidden = self.input_conv(inputs) + \
                            self.hidden_conv(prev_hidden) + \
                            self.inner_bias()
        # 分割
        i, f, c, o = tf.split(inputs_and_hidden,
                              num_or_size_splits=4, axis=-1)

        input_gate = tf.sigmoid(
            i + self.cell_product('input', prev_cell))

        forget_gate = tf.sigmoid(
            f + self.cell_product('forget', prev_cell))

        cell = forget_gate * prev_cell + input_gate * tf.tanh(c)

        output_gate = tf.sigmoid(
            o + self.cell_product('output', cell))

        hidden = output_gate * tf.tanh(cell)
        return hidden, (hidden, cell)

    # 获取卷积后输入图片流的shape
    def get_filtered_shape(self, input_size):
        rows, columns = input_size
        new_rows = np.floor((rows - 7 + 6) / 5 + 1)
        new_columns = np.floor((columns - 7 + 6) / 3 + 1)
        return [new_rows, new_columns]

    # Hadamard product for memory cell
    def cell_product(self, name, cell):
        with tf.variable_scope(name):
            weights = tf.get_variable(
                name=name,
                shape=[1] + self.image_shape + [self.output_channels],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=.1),
                regularizer=tf.contrib.layers.l2_regularizer(scale=.1))
            return cell * weights  # 广播

    # Convolution operator for hidden states
    def hidden_conv(self, hidden):
        with tf.variable_scope('hidden_conv'):
            initializers = {"w": tf.truncated_normal_initializer(stddev=.1)}
            regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=.1)}

            conv2d_35 = sntConv2D(
                output_channels=self.output_channels * 4,
                kernel_shape=[3, 5],
                stride=[1, 1],
                initializers=initializers,
                regularizers=regularizers,
                use_bias=False,
                name='conv35')
            return conv2d_35(hidden)

    # Convolution operator for inputs
    def input_conv(self, inputs):
        with tf.variable_scope('input_conv'):
            initializers = {"w": tf.truncated_normal_initializer(stddev=.1)}
            regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=.1)}

            conv2d_71 = sntConv2D(
                output_channels=CONV_SHAPE,
                kernel_shape=[7, 1],
                stride=[5, 1],
                initializers=initializers,
                regularizers=regularizers,
                use_bias=False,
                name='conv71')

            conv2d_17 = sntConv2D(
                output_channels=CONV_SHAPE,
                kernel_shape=[1, 7],
                stride=[1, 3],
                initializers=initializers,
                regularizers=regularizers,
                use_bias=False,
                name='conv17')

            conv2d_33 = sntConv2D(
                output_channels=self.output_channels * 4,
                kernel_shape=[3, 3],
                stride=[1, 1],
                initializers=initializers,
                regularizers=regularizers,
                use_bias=False,
                name='conv33')

            inputs = _swich(conv2d_17(inputs))
            inputs = _swich(conv2d_71(inputs))
            return conv2d_33(inputs)

    # bias in four equation
    def inner_bias(self):
        """gate 偏置 [1, rows, columns, output_channels*4]"""
        return tf.get_variable(
            name='bias',
            shape=[1] + self.image_shape + [self.output_channels * 4],
            dtype=tf.float32,
            initializer=tf.constant_initializer(1.))

    def get_regularizer(self):
        """正则化 —— 2范数"""
        return self.get_variables(tf.GraphKeys.REGULARIZATION_LOSSES)

    @property
    def state_size(self):
        """Returns a description of the state size, without batch dimension."""
        return (tf.TensorShape(self.image_shape + [self.output_channels]),
                tf.TensorShape(self.image_shape + [self.output_channels]))

    @property
    def output_size(self):
        """Returns a description of the output size, without batch dimension."""
        return tf.TensorShape(self.image_shape + [self.output_channels])

    def initial_state(self, batch_size, dtype):
        """Returns an initial state with zeros, for a batch size and data type.

        NOTE: This method is here only for illustrative purposes, the corresponding
        method in its superclass should be already doing this.
        """
        sz1, sz2 = self.state_size
        # Prepend batch size to the state shape, and create zeros.
        return (tf.zeros([batch_size] + sz1.as_list(), dtype=dtype),
                tf.zeros([batch_size] + sz2.as_list(), dtype=dtype))


class StackedRNN(sntRNNCore):
    def __init__(self,
                 output_channels,
                 input_size,
                 name='stack_rnn'):
        super().__init__(name=name)
        self.output_channels = output_channels
        # Multi-RNNCell
        self.R1 = Conv2DLSTM('R1', RNN_OUT_SHAPE, input_size)
        self.R1_shape = self.R1.get_filtered_shape(input_size)
        self.R2 = Conv2DLSTM('R2', output_channels, self.R1_shape)
        self.R2_shape = self.R2.get_filtered_shape(self.R1_shape)

    def _build(self, inputs, prev_state):
        h1, s1, h2, s2 = prev_state
        new_inputs, next_1 = self.R1(inputs, (h1, s1))
        output, next_2 = self.R2(new_inputs, (h2, s2))
        next_state = (next_1[0], next_1[1], next_2[0], next_2[1])
        return output, next_state

    @property
    def state_size(self):
        """Returns a description of the state size, without batch dimension."""
        return (tf.TensorShape(self.R1_shape + [RNN_OUT_SHAPE]),
                tf.TensorShape(self.R1_shape + [RNN_OUT_SHAPE]),
                tf.TensorShape(self.R2_shape + [self.output_channels]),
                tf.TensorShape(self.R2_shape + [self.output_channels]))

    @property
    def output_size(self):
        """Returns a description of the output size, without batch dimension."""
        return tf.TensorShape(self.R2_shape + [self.output_channels])

    def initial_state(self, batch_size, dtype):
        """Returns an initial state with zeros, for a batch size and data type.

        NOTE: This method is here only for illustrative purposes, the corresponding
        method in its superclass should be already doing this.
        """
        sz1, sz2, sz3, sz4 = self.state_size
        # Prepend batch size to the state shape, and create zeros.
        return (tf.zeros([batch_size] + sz1.as_list(), dtype=dtype),
                tf.zeros([batch_size] + sz2.as_list(), dtype=dtype),
                tf.zeros([batch_size] + sz3.as_list(), dtype=dtype),
                tf.zeros([batch_size] + sz4.as_list(), dtype=dtype))
