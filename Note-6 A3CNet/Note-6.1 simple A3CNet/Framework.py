import numpy as np
from ACNet import ACNet
import gym

STATE_SIZE = 4
ACTION_SIZE = 2
MAX_EPISODE_LENGTH = 1000
MAX_EPISODES = 1000
GAMMA = .99
GAME = 'CartPole-v0'


class ExplorerFramework(object):
    def __init__(self, access, name, state_size, action_size):
        self.Access = access
        self.AC = ACNet(self.Access, state_size, action_size, name)
        self.env = gym.make(GAME).unwrapped
        self.name = name

    def get_bootstrap(self, done, sess, next_state):
        if done:
            terminal = 0
        else:
            terminal = self.AC.get_value(
                sess, np.expand_dims(next_state, axis=0))[0][0]
        return terminal

    def get_output(self, sess, inputs, actions, targets):
        return self.AC.get_losses(sess, inputs, actions, targets)

    def run(self, sess, max_episodes, t_max=32):
        episode_score_list = []
        episode = 0
        while episode < max_episodes:
            episode += 1
            episode_socre = self.run_episode(sess, t_max)
            episode_score_list.append(episode_socre)
        print(episode_score_list)

    def run_episode(self, sess, t_max= 32):
        t_start = t = 0
        episode_score = 0
        buffer_state = []
        buffer_action = []
        buffer_reward = []

        self.AC.init_network(sess)
        state = self.env.reset()
        while True:
            t += 1
            action = self.AC.action_choose(sess, state)
            next_state, reward, done, info = self.env.step(action)
            # buffer for loop
            episode_score += reward
            buffer_state.append(state)
            buffer_action.append(action)
            buffer_reward.append(reward)
            state = next_state

            if t - t_start == t_max or done:
                t_start = t
                terminal = self.get_bootstrap(done, sess, next_state)

                buffer_target = []
                for r in buffer_reward[::-1]:
                    terminal = r + GAMMA * terminal
                    buffer_target.append(terminal)
                buffer_target.reverse()

                inputs = np.vstack(buffer_state)
                actions = np.squeeze(np.vstack(buffer_action), axis=1)
                targets = np.squeeze(np.vstack(buffer_target), axis=1)
                buffer_state = []
                buffer_action = []
                buffer_reward = []
                # update Access gradients
                self.AC.train_step(sess, inputs, actions, targets)
                # if self.name == 'W0':
                #    print(t, self.get_output(sess, inputs, actions, targets))

                # update local network
                self.AC.init_network(sess)
                # if self.name == 'W0':
                #    print(t, self.get_output(sess, inputs, actions, targets))

            if done or t > MAX_EPISODE_LENGTH:
                return episode_score
