import random
import tensorflow as tf
from FCNet import FCNet

LOSS_V = 10
ENTROPY_BETA = 1
_EPSILON = 1e-6
actor_learning_rate = 1e-3
critic_learning_rate = 1e-3

class ACNet(object):

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = list(range(self.action_size))

        self.inputs = tf.placeholder(tf.float32, [None, state_size], 'inputs')
        self.actions = tf.placeholder(tf.int32, [None], 'aciton') # which action was taken
        self.a_t = tf.one_hot(self.actions, self.action_size, name='action_taken')
        self.targets = tf.placeholder(tf.float32, [None], 'disounted_reward')
        # not immediate but n step discounted
        self.R = tf.expand_dims(self.targets, axis=1)


        # build network
        self.actor = FCNet('actor')
        self.critic = FCNet('critic')
        # policy and deterministic policy
        self.P = tf.nn.softmax(self.actor(self.inputs, self.action_size))
        self.DP = tf.reduce_sum(self.P * self.a_t, axis=1, keep_dims=True)
        # choose action one step, action probability
        self.AP = tf.squeeze(self.P, axis=0)
        self.log_DP = tf.log(self.DP + _EPSILON)
        # value and advantage
        self.V = self.critic(self.inputs, 1) # value predicted
        self.A = self.R - self.V
        # loss
        self.loss_policy = -tf.reduce_sum(self.log_DP * tf.stop_gradient(self.A))
        self.loss_value = LOSS_V * tf.nn.l2_loss(self.A)
        self.loss_entropy = ENTROPY_BETA * tf.reduce_sum(self.P * tf.log(self.P + _EPSILON))

        # optimizer
        self.actor_optimizer = tf.train.AdamOptimizer(
            actor_learning_rate).minimize(self.loss_policy + self.loss_entropy)
        self.critic_optimizer = tf.train.AdamOptimizer(
            critic_learning_rate).minimize(self.loss_value)

        # session
        self.sess = tf.Session()
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init_op)
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()

    def predict_value(self, state):
        return self.sess.run(self.V, {self.inputs: state})

    def predict_policy(self, state):
        return self.sess.run(self.P, {self.inputs: state})

    def predict_action(self, state):
        policy = self.sess.run(self.AP, {self.inputs: state})
        return random.choices(self.action_space, policy)[0]

    def train_actor(self, states, actions, targets):
        self.sess.run(self.actor_optimizer,
                      {self.inputs: states, self.actions: actions, self.targets: targets})

    def train_critic(self, states, targets):
        self.sess.run(self.critic_optimizer,
                      {self.inputs: states, self.targets: targets})

    def get_loss(self, states, actions, targets):
        fetches = [self.loss_policy, self.loss_value, self.loss_entropy]
        feed_dict = {self.inputs: states, self.actions: actions, self.targets: targets}
        return self.sess.run(fetches, feed_dict)