import numpy as np
import pandas as pd

_EPSILON = 1e-12

trading_fee = 0.0005

# SHARPE_WINDOW = 10
EMA_WINDOW = 10


def replace_nan(array):
    series = pd.Series(array)
    series = series.fillna(0)
    return series.values


def margin_utility_gradient(hist_ret):
    """
    Compute margin utility gradient, refer to week10-moody.ppt
    :param hist_ret: history return from (t-EMA_WINDOW-1) to (t-1)
    :return: the value
    """
    length = len(hist_ret)
    assert length > 0
    eta = 1 / length
    hist_ret_square = hist_ret ** 2
    prev_a = hist_ret.ewm(adjust=True, min_periods=length, alpha=eta).mean().iloc[-1]   # A_{t-1}
    prev_b = hist_ret_square.ewm(adjust=True, min_periods=length, alpha=eta).mean().iloc[-1]    # B_{t-1}
    last_r = hist_ret.iloc[-1]   # R_{t}

    numerator = prev_b - prev_a * last_r
    denominator = pow(prev_b - prev_a ** 2, 3 / 2)

    return numerator / denominator


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
        self.num_for_each = self.cash_for_each / (self.table_close[0] *
                                                  self.trading_info['contract_multiplier'] * self.trading_info[
                                                      'margin_rate']) / 100
        self.num_for_each = np.floor(self.num_for_each)
        self.current_total_value = self.cash * 0.99  # total_value = cash + margin + open-profit
        self.margins = self.cash / self.table_close.shape[1]  # divide equally
        self.portfolio = np.zeros(self.table_open.shape[1])  # 股票持仓数量
        self.current_position = np.zeros(self.table_close.shape[1])  # current position

    def reset(self):
        self.cash = 5e6
        self.buffer_value = []
        self.buffer_reward = []
        self.buffer_sharpe = []

        self.cash_for_each = (self.cash * 0.99) / self.stock_count  # divide equally
        self.num_for_each = self.cash_for_each / (self.table_close[0] *
                                                  self.trading_info['contract_multiplier'] * self.trading_info[
                                                      'margin_rate']) / 100
        self.num_for_each = np.floor(self.num_for_each)
        self.current_total_value = self.cash * 0.99
        self.margins = self.cash / self.table_close.shape[1]
        self.portfolio = np.zeros(self.table_open.shape[1])
        self.current_position = np.zeros(self.table_close.shape[1])

    def step(self, step_counter, action_vector):
        self.num_for_each = self.cash_for_each / (self.table_close[step_counter] *
                                                  self.trading_info['contract_multiplier'] * self.trading_info[
                                                      'margin_rate']) / 100
        self.num_for_each = np.floor(self.num_for_each)
        closes = self.table_close[step_counter]
        operator = action_vector - 1  # 0,1,2 -> -1,0,1
        # preprocess, set op to NaN for stop trading futures
        stop_trading_indexes = np.where((closes == 0) | (closes == np.nan))
        operator[stop_trading_indexes] = np.array([np.nan] * len(stop_trading_indexes))

        self.trade_to_target(operator, step_counter)

        next_total_value = self.assess(step_counter)

        value_length = len(self.buffer_value)
        if value_length <= EMA_WINDOW + 1:
            reward = 0.1
        else:
            hist_ret = pd.Series(self.buffer_value[value_length - EMA_WINDOW - 2: value_length - 1]).pct_change().dropna()
            reward = margin_utility_gradient(hist_ret)

        self.buffer_reward.append(reward)
        self.buffer_value.append(next_total_value)

        if step_counter >= self.table_open.shape[0] - 20:
            done = True
        elif self.cash < 2e6:
            done = True
        else:
            done = False

        return reward, done

    def trade_to_target(self, operator, step_counter):
        """
        Trade from current position to target position. Adjust total_value and position accordingly.
        :param operator:        new operator indicates target position
        :param step_counter:    current step counter
        :return: void
        """
        trading = operator - self.current_position
        # compute current total value as a result of tradings happened yesterday
        self.current_total_value += np.nansum(self.current_position * self.table_return[step_counter])
        # minus trading fee
        fees = abs(trading) * self.portfolio * self.trading_info['contract_multiplier'] * trading_fee
        self.current_total_value -= np.nansum(fees)
        if all(trading == [0] * len(trading)):  # position didn't change, no trading
            pass
        else:
            self.current_position = operator
            self.portfolio = self.portfolio + trading * self.num_for_each

    def assess(self, step_counter):
        next_total_value = self.current_total_value + np.nansum(self.current_position * self.table_return[step_counter + 1])
        return next_total_value


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
    high_freq = get_price(stock_list, start_date_features, end_date, fields=fields_hf, adjust_type='post',
                          frequency='15m')
    trading_info = dict()
    trading_info['margin_rate'] = [i.margin_rate for i in instruments(stock_list)]
    trading_info['contract_multiplier'] = [i.contract_multiplier for i in instruments(stock_list)]

    Q = Quotes(daily, trading_info)
    Q.step(0, np.array([1, 1, 0]))
