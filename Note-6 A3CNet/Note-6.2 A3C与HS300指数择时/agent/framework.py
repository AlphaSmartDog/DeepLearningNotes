import numpy as np
import tensorflow as tf
from agent.forward import Forward
from config import *


_EPSILON = 1e-6  # avoid nan


# local network for advantage actor-critic which are also know as A2C
class Framework(object):
    def __init__(self, access, state_size, action_size, scope_name):
        self.Access = access
        self.action_size = action_size
        self.action_space = list(range(action_size))

        with tf.variable_scope(scope_name):
            # placeholder
            self.inputs = tf.placeholder(tf.float32, [None] + state_size, "states")
            self.actions = tf.placeholder(tf.int32, [None], "actions")
            self.targets = tf.placeholder(tf.float32, [None], "discounted_rewards")

            # network interface
            self.actor = Forward('actor')
            self.critic = Forward('critic')
            self.policy = tf.nn.softmax(self.actor(self.inputs, self.action_size))
            self.value = self.critic(self.inputs, 1)
            self.policy_step = tf.squeeze(self.policy, axis=0)
            self.greedy_action = tf.argmax(self.policy_step)

            # losses
            self._build_losses()

            # async framework
            self._build_async_interface()

            self._build_interface()
            print('graph %s' % (str(scope_name)))

    def _build_losses(self):
        # value loss
        targets = tf.expand_dims(self.targets, axis=1)
        self.advantage = targets - self.value
        self.value_loss = tf.reduce_mean(tf.square(self.advantage))
        # policy loss
        action_gather = tf.one_hot(self.actions, self.action_size)
        policy_action = tf.reduce_sum(self.policy * action_gather,
                                      axis=1, keep_dims=True)
        log_policy_action = tf.log(policy_action + _EPSILON)
        self.policy_loss = -tf.reduce_mean(
            tf.stop_gradient(self.advantage) * log_policy_action)
        # entropy loss
        entropy_loss = tf.reduce_sum(
            self.policy * tf.log(self.policy + _EPSILON),
            axis=1, keep_dims=True)
        self.entropy_loss = tf.reduce_mean(entropy_loss)
        # regularization
        self.actor_norm = tf.add_n(self.actor.get_regularization()) * ACTOR_NORM_BETA
        self.critic_norm = tf.add_n(self.critic.get_regularization()) * CRITIC_NORM_BETA
        # total loss
        self.actor_loss = self.policy_loss + ENTROPY_BETA * self.entropy_loss + self.actor_norm
        self.critic_loss = self.value_loss + self.critic_norm

        # interface adjustment parameters
        self.a_actor_loss = self.actor_loss
        self.a_policy_mean = -tf.reduce_mean(log_policy_action)
        self.a_policy_loss = self.policy_loss
        self.a_entropy_loss = ENTROPY_BETA * self.entropy_loss
        self.a_actor_norm = self.actor_norm
        self.a_critic_loss = self.critic_loss
        self.a_value_loss = self.value_loss
        self.a_critic_norm = self.critic_norm
        self.a_value_mean = tf.reduce_mean(self.value)
        self.a_advantage = tf.reduce_mean(self.advantage)

    def _build_interface(self):
        self.a_interface = [self.a_actor_loss,
                            self.a_actor_grad,
                            self.a_policy_mean,
                            self.a_policy_loss,
                            self.a_entropy_loss,
                            self.a_actor_norm,
                            self.a_critic_loss,
                            self.a_critic_grad,
                            self.a_value_loss,
                            self.a_critic_norm,
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

    def get_policy(self, sess, inputs):
        return sess.run(self.policy, {self.inputs: inputs})

    def get_stochastic_action(self, sess, inputs, epsilon=0.95):
        # get stochastic action for train
        if np.random.uniform() < epsilon:
            policy = sess.run(self.policy_step,
                              {self.inputs: np.expand_dims(inputs, axis=0)})
            return np.random.choice(self.action_space, 1, p=policy)[0]
        else:
            return np.random.randint(self.action_size)

    def get_deterministic_policy_action(self, sess, inputs):
        # get deterministic action for test
        return sess.run(self.greedy_action,
                        {self.inputs: np.expand_dims(inputs, axis=0)})

    def get_value(self, sess, inputs):
        return sess.run(self.value, {self.inputs: inputs})

    def train_step(self, sess, inputs, actions, targets):
        feed_dict = {self.inputs: inputs,
                     self.actions: actions,
                     self.targets: targets}
        sess.run(self.update_global, feed_dict)

    def init_network(self, sess):
        """
        init or update local network
        :param sess:
        :return:
        """
        sess.run(self.update_local)

    def get_losses(self, sess, inputs, actions, targets):
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
                     self.targets: targets}
        return sess.run(self.a_interface, feed_dict)
