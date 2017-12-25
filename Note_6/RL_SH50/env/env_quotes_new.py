import numpy as np
import pandas as pd

_EPSILON = 1e-12


def replace_nan(array):
    series = pd.Series(array)
    series = series.fillna(0)
    return series.values

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

        self.cash_for_each = (self.cash * 0.99) / self.stock_count  # divide equally
        self.num_for_each = ([self.cash_for_each] * self.stock_count) / \
            (self.table_close[0] * self.trading_info['contract_multiplier'] * self.trading_info['margin_rate']) / 100
        self.num_for_each = replace_nan(self.num_for_each)
        self.current_total_value = 0
        self.margins = self.cash / self.table_close.shape[1]  # divide equally
        self.portfolio = np.zeros(self.table_open.shape[1])  # 股票持仓数量
        self.current_position = np.zeros(self.table_close.shape[1])  # current position

    def reset(self):
        self.cash = 5e6
        self.buffer_value = []
        self.buffer_reward = []
        self.buffer_sharpe = []

        self.cash_for_each = (self.cash * 0.99) / self.stock_count  # divide equally
        self.num_for_each = ([self.cash_for_each] * self.stock_count) / \
                            (self.table_close[0] * self.trading_info['contract_multiplier'] * self.trading_info['margin_rate']) / 100
        self.num_for_each = np.floor(replace_nan(self.num_for_each))
        self.current_total_value = 0
        self.margins = self.cash / self.table_close.shape[1]  # divide equally
        self.portfolio = np.zeros(self.table_open.shape[1])  # 股票持仓数量
        self.current_position = np.zeros(self.table_close.shape[1])  # current position

    def step(self, step_counter, action_vector):
        self.num_for_each = ([self.cash_for_each] * self.stock_count) / (self.table_close[step_counter] *
                                    self.trading_info['contract_multiplier'] * self.trading_info['margin_rate']) / 100
        self.num_for_each = np.floor(replace_nan(self.num_for_each))
        closes = self.table_close[step_counter]
        operator = action_vector - 1  # 0,1,2 -> -1,0,1
        # preprocess, set op to NaN for stop trading futures
        stop_trading_indexes = np.where((closes == 0) | (closes == np.nan))
        operator[stop_trading_indexes] = np.array([np.nan] * len(stop_trading_indexes))

        self.trade_to_target(operator, step_counter)

        next_total_value = self.assess(step_counter)
        print(next_total_value)

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
        elif self.cash < 2e6:
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
        if all(operator == self.current_position):  # position didn't change
            # no trading occur, update current and next value
            self.current_total_value += np.sum(replace_nan(operator * self.table_return[step_counter]))
        else:
            trading = operator - self.current_position
            self.current_position = operator
            self.portfolio = self.portfolio + trading * self.num_for_each
            self.current_total_value = self.current_total_value + \
                                       np.sum(replace_nan(trading * self.table_close[step_counter]))

    def assess(self, step_counter):  # next total_value
        return_of_position = self.current_position * self.table_return[step_counter + 1]
        return_of_position = replace_nan(return_of_position)
        total_value = self.current_total_value + sum(return_of_position)  # if all nan, =0
        return total_value


if __name__ == '__main__':
    import rqdatac
    from rqdatac import *
    rqdatac.init('xinjin', '123456', ('172.19.182.162', 16003))

    # get data
    # stock_list = ['000300.XSHG', '000016.XSHG', '000905.XSHG']
    stock_list = ['C99', 'CS99', 'A99']
    start_date = '2012-01-01'
    start_date_features = '2011-12-01'
    end_date = '2017-06-30'
    fields_daily = ['open', 'close']
    fields_hf = ['open', 'high', 'low', 'close', 'total_turnover']

    daily = get_price(stock_list, start_date, end_date, fields=fields_daily, adjust_type='post', frequency='1d')
    high_freq = get_price(stock_list, start_date_features, end_date, fields=fields_hf, adjust_type='post', frequency='15m')
    trading_info = dict()
    trading_info['margin_rate'] = [i.margin_rate for i in instruments(stock_list)]
    trading_info['contract_multiplier'] = [i.contract_multiplier for i in instruments(stock_list)]

    Q = Quotes(daily, trading_info)
    Q.step(0, np.array([1, 1, 0]))
