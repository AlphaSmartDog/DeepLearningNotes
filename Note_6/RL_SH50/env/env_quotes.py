import numpy as np
import pandas as pd

_EPSILON = 1e-12


class Quotes(object):
    def __init__(self, daily_prices):
        self.table_open = np.array(daily_prices.open)  # 开盘价
        self.table_close = np.array(daily_prices.close)  # 收盘价
        self.buy_free = 2.5e-4 + 1e-4
        self.sell_free = 2.5e-4 + 1e-3 + 1e-4
        self.reset()

    def reset(self):
        self.portfolio = np.zeros(self.table_open.shape[1])  # 股票持仓数量
        self.cash = 5e6
        self.valuation = 0  # 持仓估值
        self.total_value = self.cash + self.valuation
        self.buffer_value = []
        self.buffer_reward = []
        self.buffer_sharpe = []

    def buy(self, op, opens, closes):
        cash = self.cash * 0.99  # 可使用资金量
        mask = np.sign(np.maximum(opens - 1, 0))  # 掩码 去掉停盘数据
        op = mask * op
        sum_buy = np.maximum(np.sum(op), 1)
        cash_buy = op * (cash / sum_buy)  # 等资金量
        # num_buy = np.round(cash_buy / ((opens + _EPSILON) * 100))  # 手
        # 使用收盘价
        num_buy = np.round(cash_buy / ((closes + _EPSILON) * 100))  # 手
        self.cash -= np.sum(closes * 100 * num_buy * (1 + self.buy_free))  # 买入股票操作
        self.portfolio += num_buy * 100

    def sell(self, op, opens, closes):
        mask = np.sign(np.maximum(opens - 1, 0))
        num_sell = self.portfolio * op * mask  # 卖出股票数量
        self.cash -= np.sum(closes * num_sell * (1 - self.sell_free))
        self.portfolio += num_sell

    def assess(self, closes):   # 用第二天的收盘价评估 total_value
        total_value = self.cash + np.sum(self.portfolio * closes)
        return total_value

    def step(self, step_counter, action_vector):
        # 获取报价单
        opens = self.table_open[step_counter]
        closes = self.table_close[step_counter]
        # 买卖操作信号
        op = action_vector - 1  # 0,1,2 -> -1,0,1
        buy_op = np.maximum(op, 0)
        sell_op = np.minimum(op, 0)
        # 卖买操作
        self.sell(sell_op, opens, closes)
        self.buy(buy_op, opens, closes)
        # 次日估值
        next_closes = self.table_close[step_counter + 1]
        new_value = self.assess(next_closes)

        '''
        reward = np.log(new_value / self.total_value)
        self.total_value = new_value
        self.buffer_value.append(new_value)
        self.buffer_reward.append(reward)

        '''
        if step_counter <= 10:
            reward = 0.1
            new_sharpe = 0
            if step_counter == 10:
                hist_ret = pd.Series(self.buffer_value[step_counter - 10:step_counter]).pct_change().dropna()
                new_sharpe = hist_ret.mean() / (hist_ret.std() + 0.0001) * 16
        else:
            hist_ret = pd.Series(self.buffer_value[step_counter - 10:step_counter]).pct_change().dropna()
            new_sharpe = hist_ret.mean() / (hist_ret.std() + 0.0001) * 16
            print('step_counter: {}, buffer length: {}, diff: {}'
                  .format(step_counter, len(self.buffer_sharpe), step_counter - len(self.buffer_sharpe)))
            # if len(self.buffer_sharpe) != step_counter + 1 or len(self.buffer_sharpe) == 0:
            #     print(step_counter, len(self.buffer_sharpe))
            last_sharpe = self.buffer_sharpe[-1]
            reward = new_sharpe - last_sharpe

        self.buffer_reward.append(reward)
        self.buffer_value.append(new_value)
        prev_len = len(self.buffer_sharpe)
        self.buffer_sharpe.append(new_sharpe)
        current_len = len(self.buffer_sharpe)
        if current_len - prev_len != 1:
            print('previous buffer length {}, after append new sharpe, '
                  'length is {}'.format(prev_len, current_len))
        # print(len(self.buffer_sharpe))
        self.total_value = new_value

        if step_counter >= self.table_open.shape[0] - 20:
            done = True
        elif self.total_value < 2e6:
            done = True
        else:
            done = False

        return reward, done