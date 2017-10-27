import tensorflow as tf
from networks.FCNet import FCNet

LOSS_V = 1000
_EPSILON = 1e-6
actor_lr = 1e-3
critic_lr = 1e-3

class ACNet(object):
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.state = tf.placeholder(tf.float32, [None, state_size], 'state')
        self.R = tf.placeholder(tf.float32, [None, 1], 'disounted_reward')  # not immediate but n step discounted
        self.a_t = tf.placeholder(tf.int32, [None], 'aciton') # which action was taken
        self.a_t_onehot = tf.one_hot(self.a_t, action_size)

        # build network
        self.actor = FCNet('actor')
        self.critic = FCNet('critic')
        self.policy = tf.nn.softmax(self.actor(self.state, self.action_size), name='action_probability')
        self.value = self.critic(self.state, 1) # value predicted

        # advantage and losses
        self.deterministic_policy = tf.reduce_sum(self.policy * self.a_t_onehot, axis=1, keep_dims=True)
        self.log_deterministic_policy = tf.log(self.deterministic_policy + _EPSILON)
        self.advantage = self.R - self.value
        self.loss_policy = - tf.reduce_sum(self.log_deterministic_policy * tf.stop_gradient(self.advantage))
        self.loss_value = LOSS_V * tf.nn.l2_loss(self.advantage) # minimize value error
        # entropy seem wrong ?
        entropy = tf.reduce_sum(self.policy * tf.log(self.policy + _EPSILON), axis=1, keep_dims=True)
        self.entropy = tf.reduce_sum(entropy)

        # optimizer
        self.actor_optimizer = tf.train.AdamOptimizer(actor_lr).minimize(self.loss_policy + self.entropy)
        self.critic_optimizer = tf.train.AdamOptimizer(critic_lr).minimize(self.loss_value)

        # session
        self.sess = tf.Session()
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init_op)
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()

    def predict_value(self, state):
        return self.sess.run(self.value, {self.state: state})

    def predict_policy(self, state):
        return self.sess.run(self.policy, {self.state: state})

    def train_actor(self, states, actions, R):
        self.sess.run(self.actor_optimizer, {self.state: states, self.a_t: actions, self.R: R})

    def train_critic(self, states, R):
        self.sess.run(self.critic_optimizer, {self.state: states, self.R: R})



