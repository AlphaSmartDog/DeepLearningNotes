import tensorflow as tf
from Network import Network

LEARNING_RATE = 1e-3
DECAY_RATE = .99


class Access(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = list(range(action_size))

        with tf.variable_scope('Access'):
            # placeholder
            self.inputs = tf.placeholder(tf.float32, shape=[None, state_size], name="state")
            self.action = tf.placeholder(tf.int32, shape=[None], name="action")
            self.target = tf.placeholder(tf.float32, shape=[None], name="discounted_reward")
            self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")
            # network
            self.actor = Network('actor')
            self.critic = Network('critic')
            self.policy = tf.nn.softmax(self.actor(self.inputs, self.action_size))
            self.value = self.critic(self.inputs, 1)
            self.policy_step = tf.squeeze(self.policy, axis=0)
        # global optimizer
        self.optimizer_actor = tf.train.RMSPropOptimizer(
            LEARNING_RATE, DECAY_RATE, name='optimizer_actor')
        self.optimizer_critic = tf.train.RMSPropOptimizer(
            LEARNING_RATE, DECAY_RATE, name='optimizer_critic')

    def get_trainable(self):
        return [self.actor.get_variables(), self.critic.get_variables()]