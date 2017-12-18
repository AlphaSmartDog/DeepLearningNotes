import tensorflow as tf
from agent.forward import ActorCriticNet
from params import *


class Access(object):
    def __init__(self, inputs_shape, action_size):
        with tf.variable_scope('Access'):
            # placeholder
            self.inputs = tf.placeholder(
                tf.float32, [None] + inputs_shape, 'inputs')

            # neural network interface
            inputs = tf.expand_dims(self.inputs, axis=-1)
            self.net = ActorCriticNet()
            self.policy, self.value = \
                self.net(inputs, action_size)

        # global optimizer
        self.optimizer = tf.train.RMSPropOptimizer(
            LEARNING_RATE, DECAY_RATE, name='optimizer')

        # saver
        var_list = list(self.get_trainable())
        self.saver = tf.train.Saver(var_list=var_list)

    def get_trainable(self):
        return list(self.net.get_variables())

    def save(self, sess, path):
        self.saver.save(sess, path)

    def restore(self, sess, path):
        var_list = list(self.get_trainable()[0] + self.get_trainable()[1])
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, path)