import numpy as np
import gym
from ACNet_adjust import ACNet

_EPSILON = 1e-6
np.random.seed(1)
MAX_EPISODES = 1000
GAME = 'CartPole-v0'

class Agent(object):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.AC = ACNet(self.state_size, self.action_size)
        self.episode = 1
        self.accumulate_reward_list = []
        self.accumulate_reward = 0
        self.T = 0
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.R = []

    def train(self):
        states = np.vstack(self.states)
        actions = np.squeeze(np.vstack(self.actions), axis=1)
        R = np.squeeze(np.vstack(self.R), axis=1)
        self.AC.train_actor(states, actions, R)
        self.AC.train_critic(states, R)


    def run_episode(self, t_max=10):
        state = self.env.reset()
        t = 0  # initialize thread step counter t <- 1
        while True:
            action = self.AC.predict_action(np.expand_dims(state, axis=0))
            next_state, reward, done, info = self.env.step(action)

            self.accumulate_reward += reward
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)

            state = next_state
            t += 1
            if done:
                self.accumulate_reward = self.accumulate_reward * 0.9 + reward * 0.1
                R = 0
                self.update_bellman(R)
                self.train()
                break
            elif t%t_max==0:
                R = self.AC.predict_value(np.expand_dims(next_state, axis=0))
                self.update_bellman(R)
                self.train()

    def update_bellman(self, R):
        for i in range(len(self.rewards), 0, -1):
            self.R.append(self.rewards[i-1] + 0.9 * R)
            R = self.AC.predict_value([self.states[i-1]])
        self.R = np.flip(self.R, axis=0)

    def run(self, MAX_EPISODES, t_max):
        while self.episode < MAX_EPISODES:
            self.run_episode(t_max)
            self.episode += 1
            self.accumulate_reward_list.append(self.accumulate_reward)
        print('done')

    def run_adjust(self, MAX_EPISODES, t_max):
        while self.episode < MAX_EPISODES:
            self.adjust_parameters(t_max)
            self.episode += 1
            self.accumulate_reward_list.append(self.accumulate_reward)
        print('done')

    def get_loss(self):
        states = np.vstack(self.states)
        actions = np.squeeze(np.vstack(self.actions), axis=1)
        targets = np.squeeze(np.vstack(self.R), axis=1)
        outputs = self.AC.get_loss(states, actions, targets)
        self.clear()
        return outputs

    def adjust_parameters(self, t_max=10):
        state = self.env.reset()
        t = 0  # initialize thread step counter t <- 1
        while True:
            self.T += 1
            action = self.AC.predict_action(np.expand_dims(state, axis=0))
            next_state, reward, done, info = self.env.step(action)

            self.accumulate_reward += reward
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)

            state = next_state
            t += 1
            if done:
                self.accumulate_reward = self.accumulate_reward * 0.9 + reward * 0.1
                R = 0
                self.update_bellman(R)
                self.train()
                lp, lv, le = self.get_loss()
                print (self.T, lp, lv, le)
                break

            elif t%t_max==0:
                R = self.AC.predict_value(np.expand_dims(next_state, axis=0))
                self.update_bellman(R)
                self.train()
                lp, lv, le = self.get_loss()
                print (self.T, lp, lv, le)