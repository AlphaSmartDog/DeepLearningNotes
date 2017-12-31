import numpy as np
from emulator_v0.market import Market
from emulator_v0.trading import Trading


class Account(object):
    def __init__(self,
                 buy_fee= 2.5e-4+1e-4,
                 sell_fee=2.5e-4 + 1e-3 + 1e-4):
        self.T = Trading(buy_fee, sell_fee)
        self.M = Market()

    def reset(self):
        self.counter = 0
        self.T.reset()
        state, universe = self.M.reset()
        return np.expand_dims(state, axis=0), universe

    def step(self, order, universe):
        # 注意数据表格错位配置
        self.counter += 1
        reward, done, value, portfolio = \
            self.T.step(order, universe, self.counter)
        next_state, next_universe = self.M.step(self.counter)
        next_state = np.expand_dims(next_state, axis=0)
        return next_state, next_universe, reward, done, value, portfolio
