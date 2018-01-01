import numpy as np
from agent.framework import Framework
from emulator.main import Account

MAX_EPISODE_LENGTH = 1000
MAX_EPISODES = 1000
GAMMA = .9


class ExplorerFramework(object):
    def __init__(self, access, name, observation, action_size):
        self.Access = access
        self.AC = Framework(self.Access, observation, action_size, name)
        self.env = Account()
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
        episode = 0
        while episode < max_episodes:
            episode += 1
            _ = self.run_episode(sess, t_max)

    def run_episode(self, sess, t_max=32):
        t_start = t = 0
        episode_score = 0
        buffer_state = []
        buffer_action = []
        buffer_reward = []

        self.AC.init_network(sess)
        state = self.env.reset()
        while True:
            t += 1
            action = self.AC.get_stochastic_action(sess, state)
            reward, next_state, done = self.env.step(action)
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

                inputs = np.stack(buffer_state, axis=0)
                actions = np.squeeze(np.vstack(buffer_action), axis=1)
                targets = np.squeeze(np.vstack(buffer_target), axis=1)
                buffer_state = []
                buffer_action = []
                buffer_reward = []
                # update Access gradients
                self.AC.train_step(sess, inputs, actions, targets)

                # update local network
                self.AC.init_network(sess)

            if done or t > MAX_EPISODE_LENGTH:
                if self.name == 'W0':
                    outputs = tuple(self.get_output(sess, inputs, actions, targets))
                    print('actor: %f, actor_grad: %f, policy mean: %f, policy: %f, entropy: %f, actor_norm: %f, '
                          'critic: %f, critic_grad: %f, value: %f, critic_norm: %f, value_mean: %f, advantage: %f'
                          % outputs)
                return episode_score
            