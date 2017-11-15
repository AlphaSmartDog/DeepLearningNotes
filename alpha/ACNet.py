import numpy as np
import tensorflow as tf
from Network import ActorNet, CriticNet
from params import *

_EPSILON = 1e-6  # avoid nan


def batch_choice(a, p):
    action_list = [np.random.choice(a, p=i) for i in p]
    return np.array(action_list)


class Access(object):
    def __init__(self, state_size, batch_size, action_size):
        with tf.variable_scope('Access'):
            # placeholder
            self.inputs = tf.placeholder(tf.float32, [None, batch_size, state_size], 'inputs')
            # network interface
            self.actor = ActorNet('actor')
            self.critic = CriticNet('critic')

            self.policy = tf.nn.softmax(self.actor(action_size, self.inputs),)
            self.value = self.critic(self.inputs)

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


# local network for advantage actor-critic which are also know as A2C
class Agent(object):
    def __init__(self, scope_name, access, state_size, batch_size, action_size):
        self.Access = access
        self.action_size = action_size
        self.batch_size = batch_size
        # self.action_space = list(range(action_size))

        with tf.variable_scope(scope_name):
            # placeholder
            self.inputs = tf.placeholder(
                tf.float32, [None, batch_size, state_size], 'inputs')
            self.actions = tf.placeholder(
                tf.int32, [None, batch_size], "actions")
            self.targets = tf.placeholder(tf.float32, [None], "discounted_rewards")
            self.gather = tf.placeholder(tf.int32, [None], 'gather_list')

            # network and interface
            self.actor = ActorNet('actor')
            self.critic = CriticNet('critic')

            policy = self.actor(action_size, self.inputs)
            policy_gather = tf.gather(policy, self.gather)  # Be careful
            self.policy = tf.nn.softmax(policy_gather)

            value = self.critic(self.inputs)
            value_gather = tf.gather(value, self.gather)
            self.value = tf.squeeze(value_gather, axis=1)

            # interface for deterministic_policy_action
            self.policy_step = tf.nn.softmax(policy[-1])
            self.greedy_action = tf.argmax(self.policy_step, axis=1)

            # losses
            self._build_losses()

            # async framework
            self._build_async_interface()
            self._build_interface()
            print('graph %s' % (str(scope_name)))

    def _build_losses(self):
        # value loss
        self.advantage = (self.targets - self.value) * VALUE_BETA
        self.value_loss = tf.reduce_mean(tf.square(self.advantage))

        # policy loss
        action_gather = tf.one_hot(self.actions, self.action_size)
        policy_action = tf.reduce_sum(self.policy * action_gather, axis=2)
        log_policy_action = tf.log(policy_action + _EPSILON)
        advantage = tf.tanh(tf.expand_dims(self.advantage, axis=1))
        advantage = tf.stop_gradient(advantage)
        self.policy_loss = -tf.reduce_mean(advantage * log_policy_action)

        # entropy loss
        entropy_loss = tf.reduce_sum(
            self.policy * tf.log(self.policy + _EPSILON), axis=2)
        self.entropy_loss = tf.reduce_mean(entropy_loss)

        # # regularization
        # self.actor_norm = tf.add_n(self.actor.get_regularization()) * ACTOR_NORM_BETA
        # self.critic_norm = tf.add_n(self.critic.get_regularization()) * CRITIC_NORM_BETA

        # total loss
        # self.actor_loss = self.policy_loss + ENTROPY_BETA * self.entropy_loss + self.actor_norm
        # self.critic_loss = self.value_loss + self.critic_norm
        self.actor_loss = self.policy_loss + ENTROPY_BETA * self.entropy_loss
        self.critic_loss = self.value_loss

        # interface adjustment parameters
        self.a_actor_loss = self.actor_loss
        self.a_policy_mean = -tf.reduce_mean(log_policy_action)
        self.a_policy_loss = self.policy_loss
        self.a_entropy_loss = ENTROPY_BETA * self.entropy_loss
        # self.a_actor_norm = self.actor_norm
        self.a_critic_loss = self.critic_loss
        self.a_value_loss = self.value_loss
        # self.a_critic_norm = self.critic_norm
        self.a_value_mean = tf.reduce_mean(self.value)
        self.a_advantage = tf.reduce_mean(self.advantage)

    def _build_interface(self):
        self.a_interface = [self.a_actor_loss,
                            self.a_actor_grad,
                            self.a_policy_mean,
                            self.a_policy_loss,
                            self.a_entropy_loss,
                            self.a_critic_loss,
                            self.a_critic_grad,
                            self.a_value_loss,
                            self.a_value_mean,
                            self.a_advantage]

    def _build_async_interface(self):
        global_actor_params, global_critic_params = self.Access.get_trainable()
        local_actor_params, local_critic_params = self.get_trainable()
        actor_grads = tf.gradients(self.actor_loss, list(local_actor_params))
        critic_grads = tf.gradients(self.critic_loss, list(local_critic_params))

        # Set up optimizer with global norm clipping.
        actor_grads, self.a_actor_grad = tf.clip_by_global_norm(actor_grads, MAX_GRAD_NORM)
        critic_grads, self.a_critic_grad = tf.clip_by_global_norm(critic_grads, MAX_GRAD_NORM)

        # update Access
        actor_apply = self.Access.optimizer_actor.apply_gradients(
            zip(list(actor_grads), list(global_actor_params)))
        critic_apply = self.Access.optimizer_critic.apply_gradients(
            zip(list(critic_grads), list(global_critic_params)))
        self.update_global = [actor_apply, critic_apply]

        # update ACNet
        assign_list = []
        for gv, lv in zip(global_actor_params, local_actor_params):
            assign_list.append(tf.assign(lv, gv))
        for gv, lv in zip(global_critic_params, local_critic_params):
            assign_list.append(tf.assign(lv, gv))
        self.update_local = assign_list

    def get_trainable(self):
        return [self.actor.get_variables(), self.critic.get_variables()]

    def get_policy(self, sess, inputs, gather_list):
        return sess.run(self.policy,
                        {self.inputs: inputs, self.gather: gather_list})

    def get_value(self, sess, inputs, gather_list):
        return sess.run(self.value,
                        {self.inputs: inputs, self.gather: gather_list})

    def get_stochastic_action(self, sess, inputs, epsilon=0.95):
        # get stochastic action for train
        if np.random.uniform() < epsilon:
            policy = sess.run(self.policy_step, {self.inputs: inputs})
            return batch_choice(self.action_size, policy)
        else:
            return np.random.randint(self.action_size, size=self.batch_size)

    def get_deterministic_policy_action(self, sess, inputs):
        # get deterministic action for test
        return sess.run(self.greedy_action, {self.inputs: inputs})

    def train_step(self, sess, inputs, actions, targets, gather_list):
        feed_dict = {self.inputs: inputs,
                     self.actions: actions,
                     self.targets: targets,
                     self.gather: gather_list}
        sess.run(self.update_global, feed_dict)

    def init_network(self, sess):
        """
        init or update local network
        :param sess:
        :return:
        """
        sess.run(self.update_local)

    def get_losses(self, sess, inputs, actions, targets, gather):
        """
        get all loss functions of network
        :param sess:
        :param inputs:
        :param actions:
        :param targets:
        :return:
        """
        feed_dict = {self.inputs: inputs,
                     self.actions: actions,
                     self.targets: targets,
                     self.gather: gather}
        return sess.run(self.a_interface, feed_dict)
