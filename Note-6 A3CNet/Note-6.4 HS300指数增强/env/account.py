import numpy as np
import pandas as pd
import h5py


_EPSILON = 1e-12


access = h5py.File('env/env_access.hdf5')
fac = access['fac']
quotes = pd.read_hdf('env/day_quotes.h5')
universe = pd.read_hdf('env/universe.h5')
mask = pd.read_hdf('env/mask.hdf5').T

op_mask = pd.DataFrame(index=mask.index, columns=universe, data=0)
for i in op_mask.index:
    flags = mask.loc[i]
    op_mask.loc[i, flags] = 1

op_mask = np.array(op_mask)


def df_preprocessing(tmp):
    tmp = tmp.T
    tmp.sort_index(inplace=True)
    tmp = tmp.T
    tmp.sort_index(inplace=True)
    # 处理由指数成分变化 inf nan
    tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
    tmp.fillna(0, inplace=True)
    return tmp


opens = df_preprocessing(quotes['open'])
closes = df_preprocessing(quotes['close'])
indexes = closes.index


class Quotes(object):
    def __init__(self):
        self.table_open = np.array(opens)  # 开盘价
        self.table_close = np.array(closes)  # 收盘价
        self.buy_free = 2.5e-4 + 1e-4
        self.sell_free = 2.5e-4 + 1e-3 + 1e-4
        self.reset()

    def reset(self):
        self.portfolio = np.zeros_like(self.table_close[26])  # 股票持仓数量
        self.cash = 5e7
        self.valuation = 0  # 持仓估值
        self.total_value = self.cash + self.valuation
        self.buffer_value = []
        self.buffer_reward = []

    def buy(self, op, opens):
        cash = self.cash * 0.8  # 可使用资金量
        mask = np.sign(np.maximum(opens - 1, 0))  # 掩码 去掉停盘数据
        op = mask * op
        sum_buy = np.maximum(np.sum(op), 15)
        cash_buy = op * (cash / sum_buy)  # 等资金量
        num_buy = np.round(cash_buy / ((opens + _EPSILON) * 100))  # 手
        self.cash -= np.sum(opens * 100 * num_buy * (1 + self.buy_free))  # 买入股票操作
        self.portfolio += num_buy * 100

    def sell(self, op, opens):
        mask = np.sign(np.maximum(opens - 1, 0))
        num_sell = self.portfolio * op * mask  # 卖出股票数量
        self.cash -= np.sum(opens * num_sell * (1 - self.sell_free))
        self.portfolio += num_sell

    def assess(self, closes):
        total_value = self.cash + np.sum(self.portfolio * closes)
        return total_value

    def step(self, step_counter, op):
        # 获取报价单
        opens = self.table_open[step_counter]
        closes = self.table_close[step_counter]
        # 买卖操作信号
        buy_op = np.maximum(op, 0)
        sell_op = np.minimum(op, 0)
        # 卖买操作
        self.sell(sell_op, opens)
        self.buy(buy_op, opens)
        # 当日估值
        new_value = self.assess(closes)
        reward = np.log(new_value / self.total_value)
        self.total_value = new_value
        self.buffer_value.append(new_value)
        self.buffer_reward.append(reward)

        if step_counter > 465:
            done = True
        elif self.total_value < 4.5e7:
            done = True
        else:
            done = False

        return reward, done


class Account(object):
    def __init__(self):
        self.quote = Quotes()
        self.fac = fac
        self.mask = op_mask
        self.reset()

    def reset(self):
        self.quote.reset()
        self.step_counter = 0

        date = self.step_counter + 26  # T
        prev_date = date - 16  # T-16
        return self.fac[prev_date:date]

    def step(self, actions):
        # adjust index and op
        date = self.step_counter + 26  # T
        next_date = date + 1
        prev_date = next_date - 16  # T-16

        actions = (actions - 1) * self.mask[date]
        reward, done = self.quote.step(date, actions)
        reward *= 1e2  # 百分之一 基点

        next_state = self.fac[prev_date:next_date]
        self.step_counter += 1
        return next_state, reward, done

    def plot_data(self):
        value = self.quote.buffer_value
        reward = self.quote.buffer_reward
        return value, reward