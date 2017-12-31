import numpy as np
import pandas as pd

_EPSILON = 1e-6

# 交易日历
tradeDays = pd.read_hdf("emulator_v0/tradeDays.h5").iloc[22:]
tradeDays.reset_index(drop=True, inplace=True)
# 股票池
dataset_universe = pd.read_hdf("emulator_v0/universe_SH50.h5")
# 报价单
dataset_open = pd.read_hdf("emulator_v0/dataset_open.h5")
dataset_open.fillna(0, inplace=True)
dataset_close = pd.read_hdf("emulator_v0/dataset_close.h5")
dataset_close.fillna(0, inplace=True)
dataset_universe = pd.DataFrame(
    data=-1, index=dataset_open.index,
    columns=dataset_open.columns)


class Trading(object):
    def __init__(self, buy_fee, sell_fee):
        self.tradeDays = tradeDays
        self.universe = dataset_universe
        self.open = np.array(dataset_open)
        self.close = np.array(dataset_close)
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee
        self.reset()

    def reset(self):
        self.portfolio = np.zeros_like(self.universe.iloc[0], dtype="float")
        self.cash = 1e8
        self.valuation = 0  # 持仓估值
        self.total_value = self.cash + self.valuation

    def step(self, order, universe, counter):
        # 转换交易指令
        day = self.tradeDays[counter]
        order = order - 1
        order = self.get_order(order, universe, day)

        # 买卖操作信号
        mask = np.sign(np.maximum(self.open[counter] - 1, 0))
        buy_op = np.maximum(order, 0) * mask
        sell_op = np.minimum(order, 0) * mask

        # 卖买交易
        self.sell(sell_op, counter)
        self.buy(buy_op, counter)

        # 当日估值
        new_value = self.assess(counter)
        reward = np.log(new_value / self.total_value)
        self.total_value = new_value

        # MAX EPISODE
        if counter >= 468:
            done = True
        elif self.total_value < 1e8 * 0.2:
            done = True
        else:
            done = False
        return reward, done, self.total_value, self.portfolio

    def get_order(self, order, universe, day):
        self.universe.loc[day, universe] = order
        return np.array(dataset_universe.loc[day])

    def sell(self, op, counter):
        opens = self.open[counter]
        num_sell = self.portfolio * op
        self.cash -= np.sum(opens * num_sell * (1 - self.sell_fee))
        self.portfolio += num_sell

    def buy(self, op, counter):
        if self.cash <= 1e8 * 0.2:
            pass
        else:
            opens = self.open[counter]
            cash = self.cash * 0.9  # 可使用资金量
            buy_limit = self.portfolio * op - 1e8 * 0.05
            op = np.sign(np.minimum(buy_limit, 0)) * -1
            sum_buy = np.maximum(np.sum(op), 10)
            cash_buy = op * (cash / sum_buy)  # 等资金量
            num_buy = np.round(cash_buy / ((opens + _EPSILON) * 100)) * 100
            mask = np.sign(np.minimum(num_buy - 1e9, 0)) * -1
            num_buy = num_buy * mask
            self.cash -= np.sum(opens * num_buy * (1 + self.buy_fee))
            self.portfolio += num_buy

    def assess(self, step):
        total_value = self.cash + np.sum(self.portfolio * self.close[step])
        return total_value
