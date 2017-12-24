import numpy as np
import pandas as pd

_EPSILON = 1e-12


class Quotes(object):
    def __init__(self, daily_prices, trading_info):
        self.table_open = np.array(daily_prices.open)  # open
        self.table_close = np.array(daily_prices.close)  # close
        self.stock_count = self.table_close.shape[1]
        self.table_return = np.array(daily_prices.close.shift(1) / daily_prices.close)
        self.trading_info = trading_info

        self.cash = 5e6
        self.buffer_value = []
        self.buffer_reward = []
        self.buffer_sharpe = []

        cash_for_each = (self.cash * 0.99) / self.stock_count  # divide equally
        self.num_for_each = ([cash_for_each] * self.stock_count) / \
                            (self.table_close[0] * self.trading_info.multiplier * self.trading_info.margin) / 100
        self.current_total_value = 0
        self.margins = self.cash / self.table_close.shape[1]  # divide equally
        self.portfolio = np.zeros(self.table_open.shape[1])  # 股票持仓数量
        self.current_position = np.zeros(self.table_close.shape[1])  # current position

    def reset(self):
        self.cash = 5e6
        self.buffer_value = []
        self.buffer_reward = []
        self.buffer_sharpe = []

        self.current_total_value = 0
        self.portfolio = np.zeros(self.table_open.shape[1])  # 股票持仓数量
        self.current_position = np.zeros(self.table_close.shape[1])  # current position

    def step(self, step_counter, action_vector):
        closes = self.table_close[step_counter]
        operator = action_vector - 1  # 0,1,2 -> -1,0,1
        # preprocess, set op to NaN for stop trading futures
        stop_trading_indexes = np.where((closes == 0) | (closes == np.nan))
        operator[stop_trading_indexes] = np.array([np.nan] * len(stop_trading_indexes))

        self.trade_to_target(operator, step_counter)

        next_total_value = self.assess(step_counter)

        if step_counter <= 10:
            reward = 0.1
            new_sharpe = 0
            if step_counter == 10:
                hist_ret = pd.Series(self.buffer_value[step_counter - 10:step_counter]).pct_change().dropna()
                new_sharpe = hist_ret.mean() / (hist_ret.std() + 0.0001) * 16
        else:
            hist_ret = pd.Series(self.buffer_value[step_counter - 10:step_counter]).pct_change().dropna()
            new_sharpe = hist_ret.mean() / (hist_ret.std() + 0.0001) * 16
            if step_counter != 0 and len(self.buffer_sharpe) == 0:
                print('step_counter: {}, buffer length: {}, diff: {}'
                      .format(step_counter, len(self.buffer_sharpe), step_counter - len(self.buffer_sharpe)))
                print('portfolio: {}, position: {}'.format(self.portfolio, self.current_position))
                # last_sharpe = 0
            last_sharpe = self.buffer_sharpe[-1]
            reward = new_sharpe - last_sharpe

        self.buffer_reward.append(reward)
        self.buffer_value.append(next_total_value)
        self.buffer_sharpe.append(new_sharpe)

        if step_counter >= self.table_open.shape[0] - 20:
            done = True
        elif self.current_total_value < 2e6:
            done = True
        else:
            done = False

        return reward, done

    # TODO: adjust cash accordingly
    def trade_to_target(self, operator, step_counter):
        """
        Trade from current position to target position. Adjust total_value and position accordingly.
        :param operator:        new operator indicates target position
        :param step_counter:    current step counter
        :return: void
        """
        if operator == self.current_position:  # position didn't change
            # no trading occur, update current and next value
            self.current_total_value += np.sum(operator * self.table_return)
        else:
            trading = operator - self.current_position
            self.current_position = operator
            self.portfolio = self.portfolio + trading * self.num_for_each
            self.current_total_value = self.current_total_value + trading * self.table_close[step_counter]

    def assess(self, step_counter):  # next total_value
        total_value = self.current_total_value + np.sum(self.current_position * self.table_return[step_counter + 1])
        return total_value
