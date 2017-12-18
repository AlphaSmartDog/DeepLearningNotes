import numpy as np
import tensorflow as tf
from agent.forward import ActorCriticNet
from params import *


def batch_choice(a, p):
    action_list = [np.random.choice(a, p=i) for i in p]
    return np.array(action_list)


# local network for advantage actor-critic which are also know as A2C
class Agent(object):
    def __init__(self, name, access, inputs_shape, action_size):
        self.Access = access
        self.action_size = action_size
        batch_size = inputs_shape[0]
        self.batch_size = batch_size

        with tf.variable_scope(name):
            # placeholder
            # [Time, Batch, Rows, Columns, Channels]
            self.inputs = tf.placeholder(
                tf.float32, [None] + inputs_shape, 'inputs')
            # fix
            inputs = tf.expand_dims(self.inputs, axis=-1)
            # [T_MAX, Batch]
            self.actions = tf.placeholder(
                tf.int32, [None, batch_size], "actions")
            # [T_MAX]
            self.targets = tf.placeholder(
                tf.float32, [None], "discounted_rewards")
            self.gathers = tf.placeholder(
                tf.int32, [None], 'gather_list')

            # build network and adjust output probability
            self.net = ActorCriticNet('AC-' + name)
            policy, value = self.net(inputs, action_size)
            policy = tf.clip_by_value(
                policy, CLIP_MIN, CLIP_MAX, 'constraint')

            # interface gather and step
            # [Time, Batch, action_size] -> [T_MAX, Batch, action_size]
            self.policy = tf.gather(policy, self.gathers)
            self.value = tf.gather(value, self.gathers)  # [T_MAX]
            self.value = tf.squeeze(self.value, axis=1)
            self.policy_step = policy[-1]  # [Batch, action_size]
            self.value_step = value[-1]  # 1

            # build other function
            self._build_losses()
            self._build_async_swap()
            self._build_interface()

            print('graph %s' % (str(name)))

    def _build_losses(self):
        # value loss
        self.advantage = self.targets - self.value  # [T_MAX]
        value_loss = 0.5 * tf.square(self.advantage)

        # policy loss
        # [T_MAX, Batch, action_size] -> [T_MAX, Batch]
        policy_action = tf.reduce_sum(
            self.policy * tf.one_hot(self.actions, self.action_size), axis=2)
        # [T_MAX, Batch]
        policy_loss = -tf.log(policy_action) * tf.stop_gradient(
            tf.expand_dims(self.advantage, axis=1))
        # entropy loss [T_MAX, Batch]
        entropy_loss = tf.reduce_sum(self.policy * tf.log(self.policy), axis=2)

        # total loss
        self.critic_loss = tf.reduce_mean(value_loss)
        self.actor_loss = tf.reduce_mean(policy_loss + entropy_loss * ENTROPY_BETA)
        self.total_loss = self.critic_loss + self.actor_loss

        # interface
        self.a_total_loss = self.total_loss
        self.a_entropy_loss = tf.reduce_mean(entropy_loss)
        self.a_policy_loss = tf.reduce_mean(policy_loss)
        self.a_value_loss = tf.reduce_mean(value_loss)
        self.a_critic_loss = self.critic_loss
        self.a_actor_loss = self.actor_loss
        self.a_advantage = tf.reduce_mean(self.advantage)
        self.a_value_mean = tf.reduce_mean(self.value)
        self.a_policy_mean = tf.reduce_mean(self.policy)

    def _build_async_swap(self):
        # Get gradients from local network using local losses
        local_vars = self.get_trainable()
        self.gradients = tf.gradients(self.total_loss, local_vars)

        # Clip gradients
        grads, self.global_norm = tf.clip_by_global_norm(
            self.gradients, MAX_GRAD_NORM)

        # Update global network
        # Apply local gradients to global network
        global_vars = self.Access.get_trainable()

        self.update_global = self.Access.optimizer.apply_gradients(
            zip(grads, global_vars))

        # Update local network
        assign_list = []
        for gv, lv in zip(global_vars, local_vars):
            assign_list.append(tf.assign(lv, gv))
        self.update_local = assign_list

    def _build_interface(self):
        self.a_interface = [self.a_total_loss,
                            self.a_entropy_loss,
                            self.a_policy_loss,
                            self.a_value_loss,
                            self.a_actor_loss,
                            self.a_critic_loss,
                            self.a_advantage,
                            self.a_value_mean,
                            self.a_policy_mean,
                            self.a_advantage]

    def get_trainable(self):
        return list(self.net.get_variables())

    def init_or_update_local(self, sess):
        sess.run(self.update_local)

    def get_step_policy(self, sess, inputs):
        return sess.run(self.policy_step, {self.inputs: inputs})

    def get_step_value(self, sess, inputs):
        return sess.run(self.value_step, {self.inputs: inputs})

    def get_losses(self, sess, inputs, actions, targets, gather_list):
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
                     self.gathers: gather_list}
        return sess.run(self.a_interface, feed_dict)

    def train_step(self, sess, inputs, actions, targets, gathers):
        feed_dict = {self.inputs: inputs,
                     self.actions: actions,
                     self.targets: targets,
                     self.gathers: gathers}
        sess.run(self.update_global, feed_dict)

    # get stochastic action for train
    def get_stochastic_action(self, sess, inputs, epsilon=0.9):
        if np.random.uniform() < epsilon:
            policy = sess.run(self.policy_step, {self.inputs: inputs})
            return batch_choice(self.action_size, policy)
        else:
            return np.random.randint(self.action_size, size=self.batch_size)

    # get deterministic action for test
    def get_deterministic_policy_action(self, sess, inputs):
        policy_step = sess.run(self.policy_step, {self.inputs: inputs})
        return np.argmax(policy_step, axis=1)
