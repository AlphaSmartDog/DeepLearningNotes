import tensorflow as tf
from agent.forward import Forward
from config import *


class Access(object):
    def __init__(self, state_size, action_size):
        with tf.variable_scope('Access'):
            # placeholder
            self.inputs = tf.placeholder(tf.float32, [None] + state_size, "states")
            self.actions = tf.placeholder(tf.int32, [None], "actions")
            self.targets = tf.placeholder(tf.float32, [None], "discounted_rewards")

            # network interface
            self.actor = Forward('actor')
            self.critic = Forward('critic')
            self.policy = tf.nn.softmax(self.actor(self.inputs, action_size))
            self.value = self.critic(self.inputs, 1)

        # global optimizer
        self.optimizer_actor = tf.train.RMSPropOptimizer(
            LEARNING_RATE, DECAY_RATE, name='optimizer_actor')
        self.optimizer_critic = tf.train.RMSPropOptimizer(
            LEARNING_RATE, DECAY_RATE, name='optimizer_critic')

        # saver
        var_list = self.get_trainable()
        var_list = list(var_list[0] + var_list[1])
        self.saver = tf.train.Saver(var_list=var_list)

    def get_trainable(self):
        return [self.actor.get_variables(), self.critic.get_variables()]

    def save(self, sess, path):
        self.saver.save(sess, path)

    def restore(self, sess, path):
        var_list = list(self.get_trainable()[0] + self.get_trainable()[1])
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, path)

