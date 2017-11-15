import numpy as np
import pandas as pd


_EPSILON = 1e-12


SH50 = pd.read_csv('env03/2017_SH50.csv')
SH50.drop('Unnamed: 0', axis=1, inplace=True)
SH50.sort_values(['tradeDate', 'secID'], inplace=True)
universe = list(set(SH50['secID'].tolist()))
universe.sort()
tradeDays = list(set(SH50.tradeDate.tolist()))
tradeDays.sort()

open_list = []
close_list = []

for i in tradeDays:
    tmp = SH50.loc[SH50['tradeDate'] == i, ['secID', 'openPrice', 'closePrice']]
    tmp.set_index('secID', inplace=True)
    open_list.append(tmp['openPrice'])
    close_list.append(tmp['closePrice'])

tables_open = pd.concat(open_list, axis=1)
tables_open.columns = tradeDays
table_open = np.array(tables_open.T)

tables_close = pd.concat(close_list, axis=1)
tables_close.columns = tradeDays
table_close = np.array(tables_close.T)


class Quotes(object):
    def __init__(self):
        self.table_open = table_open  # 开盘价
        self.table_close = table_close  # 收盘价
        self.free = 1e-3 + 0.25e-3 + 0.5e-3
        self.reset()

    def reset(self):
        self.portfolio = np.zeros(50)  # 股票持仓数量
        self.cash = 5e6
        self.valuation = 0  # 持仓估值
        self.total_value = self.cash + self.valuation
        self.buffer_value = []
        self.buffer_reward = []

    def buy(self, op, opens):
        cash = self.cash * 0.8  # 可使用资金量
        mask = np.sign(np.maximum(opens - 1, 0))
        op = mask * op
        sum_buy = np.maximum(np.sum(op), 15)
        cash_buy = op * (cash / sum_buy)  # 等资金量
        num_buy = np.round(cash_buy / ((opens + _EPSILON) * 100))
        self.cash -= np.sum(opens * 100 * num_buy * (1 + self.free))  # 买入股票操作
        self.portfolio += num_buy

    def sell(self, op, opens):
        mask = np.sign(np.maximum(opens - 1, 0))
        num_sell = self.portfolio * op  # 卖出股票数量
        self.cash -= np.sum(opens * mask * 100 * num_sell * (1 - self.free))
        self.portfolio += num_sell

    def assess(self, opens, closes):
        mask = np.sign(np.maximum(opens - 1, 0))
        closes = mask * closes
        total_value = self.cash + np.sum(self.portfolio * closes * 100)
        return total_value

    def step(self, step_counter, action_vector):
        # 获取报价单
        opens = self.table_open[step_counter]
        closes = self.table_close[step_counter]
        # 买卖操作信号
        op = action_vector - 1
        buy_op = np.maximum(op, 0)
        sell_op = np.minimum(op, 0)
        # 卖买操作
        self.sell(sell_op, opens)
        self.buy(buy_op, opens)
        # 当日估值
        new_value = self.assess(opens, closes)
        reward = np.log(new_value / self.total_value)
        self.total_value = new_value
        self.buffer_value.append(new_value)
        self.buffer_reward.append(reward)

        if step_counter > 200:
            done = True
        elif self.total_value < 4e6:
            done = True
        else:
            done = False

        return reward, done