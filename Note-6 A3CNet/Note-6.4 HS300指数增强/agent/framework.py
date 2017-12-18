import numpy as np
from env.main import Account
from agent.agent import Agent
from params import *


def batch_stack(inputs):
    # gather index
    gather_list = 15 + np.arange(len(inputs))
    # stack
    a = [inputs[0][:-1]]
    b = [i[-1] for i in inputs]
    b = [np.expand_dims(i, axis=0) for i in b]
    return np.vstack(a + b), gather_list


class Framework(object):
    def __init__(self, name, access, inputs_shape, action_size):
        self.Access = access
        self.AC = Agent(name, access, inputs_shape, action_size)
        self.env = Account()
        self.name = name

    def run(self, sess, max_episodes, t_max=8):
        buffer_score = []
        buffer_loss = []
        episode = 0
        while episode < max_episodes:
            print(episode)
            episode += 1
            episode_score, outputs = self.run_episode(sess, t_max)
            buffer_score.append(episode_score)
            buffer_loss.append(outputs)
        return buffer_score, buffer_loss

    def run_episode(self, sess, t_max=8):
        t_start = t = 0
        episode_score = 1
        buffer_state = []
        buffer_action = []
        buffer_reward = []

        self.AC.init_or_update_local(sess)
        state = self.env.reset()
        while True:
            t += 1
            action = self.AC.get_stochastic_action(sess, state)
            next_state, reward, done = self.env.step(action)

            # buffer for loop
            episode_score *= (1 + reward / 100)
            buffer_state.append(state)
            buffer_action.append(action)
            buffer_reward.append(reward)
            state = next_state

            if t - t_start == t_max or done:
                t_start = t
                terminal = self.get_bootstrap(sess, next_state, done)

                buffer_target = []
                for r in buffer_reward[::-1]:
                    terminal = r + GAMMA * terminal
                    buffer_target.append(terminal)
                buffer_target.reverse()

                # stack
                inputs, gather_list = batch_stack(buffer_state)
                actions = np.vstack(buffer_action)
                targets = np.squeeze(np.vstack(buffer_target), axis=1)

                # empty buffer
                buffer_state = []
                buffer_action = []
                buffer_reward = []

                # update Access gradients
                self.AC.train_step(sess, inputs, actions, targets, gather_list)

                # update local network
                self.AC.init_or_update_local(sess)

            if done or t > MAX_EPISODE_LENGTH:
                outputs = self.get_losses(sess, inputs, actions, targets, gather_list)
                outputs = tuple(outputs)
                if self.name == 'W0':
                    print('total_loss: %f, '
                          'entropy_loss: %f, '
                          'policy_loss: %f, '
                          'value_loss: %f, '
                          'actor_loss: %f, '
                          'critic_loss: %f, '
                          'advantage: %f, '
                          'value_mean: %f, '
                          'policy_mean: %f, '
                          'advantage: %f'
                          % outputs)
                return episode_score, outputs

    def get_bootstrap(self, sess, next_state, done):
        if done:
            terminal = 0
        else:
            terminal = self.AC.get_step_value(sess, next_state)
        return terminal

    def get_losses(self, sess, inputs, actions, targets, gather_list):
        return self.AC.get_losses(sess, inputs, actions, targets, gather_list)