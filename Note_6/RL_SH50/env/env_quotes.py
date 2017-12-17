import numpy as np
import pandas as pd

_EPSILON = 1e-12

class Quotes(object):
    def __init__(self, daily_prices):
        self.table_open = np.array(daily_prices.open)  # 开盘价
        self.table_close = np.array(daily_prices.close)  # 收盘价
        self.buy_free = 2.5e-4 + 1e-4
        self.sell_free = 2.5e-4 + 1e-3 + 1e-4
        self.sell_short_margin = 0.05
        # initialize in init
        self.portfolio = np.zeros(self.table_open.shape[1])  # 股票持仓数量
        self.current_position = np.zeros(self.table_close.shape[1])  # current position
        self.sell_short_closes = np.zeros(self.table_close.shape[1])  # sell short price
        self.cash = 5e6
        self.valuation = 0  # 持仓估值
        self.total_value = self.cash + self.valuation
        self.buffer_value = []
        self.buffer_reward = []
        self.buffer_sharpe = []

    def reset(self):
        self.portfolio = np.zeros(self.table_open.shape[1])  # 股票持仓数量
        self.current_position = np.zeros(self.table_close.shape[1])  # current position
        self.sell_short_closes = np.zeros(self.table_close.shape[1])  # sell short price
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
        # use close price
        num_buy = np.round(cash_buy / ((closes + _EPSILON) * 100))  # 手
        self.cash -= np.sum(closes * 100 * num_buy * (1 + self.buy_free))  # 买入股票操作
        self.portfolio += num_buy * 100

    def sell(self, op, opens, closes):
        mask = np.sign(np.maximum(opens - 1, 0))
        num_sell = self.portfolio * op * mask  # 卖出股票数量
        self.cash -= np.sum(closes * num_sell * (1 - self.sell_free))
        self.portfolio += num_sell

    def assess(self, closes):  # 用第二天的收盘价评估 total_value
        total_value = self.cash + np.sum(self.portfolio * closes)
        return total_value

    def assess_by_position(self, trade_type_list, closes):
        return self.get_cash_by_position(trade_type_list, closes, self.cash)

    def get_cash_by_position(self, trade_type_list, closes, cash):
        """
        Compute new cash by position and closes in concern.
        :param trade_type_list: list of trade type indexes
        :param closes:          close price for each future
        :param cash:            original cash
        :return: new cash
        """
        for i in range(len(trade_type_list)):
            indexes = trade_type_list[i]  # e.g [0, 2], indicates witch future is trading in this type
            if i == 0:  # sell_close
                close_prices = closes[indexes]  # e.g [4312.3, 5654.2]
                sell_short_close_prices = self.sell_short_closes[indexes]
                cash += sum(sell_short_close_prices - close_prices)
            if i == 1:  # buy_close
                close_prices = closes[indexes]
                cash += sum(close_prices)
            if i == 2:  # sell_open
                self.sell_short_closes[indexes] = closes[indexes]
            if i == 3:  # buy_open
                close_prices = closes[indexes]
                cash -= sum(close_prices)

        return cash

    def position_to_trade_type(self, position, op):
        """
        Collect all trading types' indexes.
        :param position:    current position
        :param op:          new position
        :return: trade type list, e.g [[0, 2], [1, 3], [4], [5, 6]] (if has 7 futures)
        """
        buy_open_indexes = np.where(True == ((position == 0) & (op == 1)))
        sell_close_indexes = np.where(True == ((position == -1) & (op == 0)))
        # keep_indexes = np.where((position == op) == True)
        sell_open_indexes = np.where(True == ((position == 0) & (op == -1)))
        buy_close_indexes = np.where(True == ((position == 1) & (op == 0)))
        # put all trading indexes into an ordered list, indicates the order to perform trading
        return [sell_close_indexes, buy_close_indexes, sell_open_indexes, buy_open_indexes]

    def trade_to_target(self, trade_type_list, closes):
        """
        Trade from current position to target position.
        :param trade_type_list: trade type indexes
        :param closes:          close prices for each futures
        :return: void
        """
        cash = self.cash * 0.99  # 可使用资金量
        # TODO: adjust trading number by cash, currently only trade for one
        self.cash = self.get_cash_by_position(trade_type_list, closes, self.cash)

    def step(self, step_counter, action_vector):
        # 获取报价单
        closes = self.table_close[step_counter]
        # 买卖操作信号
        op = action_vector - 1  # 0,1,2 -> -1,0,1
        # preprocess, set op to NaN for stop trading futures
        stop_trading_indexes = np.where((closes == 0) | (closes == np.nan))
        op[stop_trading_indexes] = np.array([np.nan] * len(stop_trading_indexes))
        trade_type_list = self.position_to_trade_type(self.current_position, op)
        # current position -> target position
        self.trade_to_target(trade_type_list, closes)
        '''
        buy_op = np.maximum(op, 0)
        sell_op = np.minimum(op, 0)
        # 卖买操作
        self.sell(sell_op, opens, closes)
        self.buy(buy_op, opens, closes)
        '''
        # 次日估值
        next_closes = self.table_close[step_counter + 1]
        # new_value = self.assess(next_closes)
        new_value = self.assess_by_position(trade_type_list, next_closes)
        # update current position
        self.current_position = op

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
            if step_counter != 0 and len(self.buffer_sharpe) == 0:
                print('step_counter: {}, buffer length: {}, diff: {}'
                      .format(step_counter, len(self.buffer_sharpe), step_counter - len(self.buffer_sharpe)))
                print('portfolio: {}, position: {}'.format(self.portfolio, self.current_position))
                # last_sharpe = 0
            last_sharpe = self.buffer_sharpe[-1]
            reward = new_sharpe - last_sharpe

        self.buffer_reward.append(reward)
        self.buffer_value.append(new_value)
        self.buffer_sharpe.append(new_sharpe)
        self.total_value = new_value

        if step_counter >= self.table_open.shape[0] - 20:
            done = True
        elif self.total_value < 2e6:
            done = True
        else:
            done = False

        return reward, done
