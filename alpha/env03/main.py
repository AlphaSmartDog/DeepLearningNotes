from env03.env_market import Market
from env03.env_quotes import Quotes


class Account(object):
    def __init__(self):
        self.quote = Quotes()
        self.fac = Market()
        self.step_counter = 0

    def reset(self):
        self.quote.reset()
        self.step_counter = 0
        return self.fac.step(0)

    def step(self, actions):
        reward, done = self.quote.step(self.step_counter, actions)
        self.step_counter += 1
        next_state = self.fac.step(self.step_counter)
        return next_state, reward, done

    def plot_data(self):
        value = self.quote.buffer_value
        reward = self.quote.buffer_reward
        return value, reward
