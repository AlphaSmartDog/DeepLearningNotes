from env.env_market import Market
from env.env_quotes_new import Quotes
from env.env_factor import get_factor_array

DAY_SHIFT = 60

class Account(object):
    def __init__(self, daily_prices, high_freq_data, trading_info):
        # 日线数据
        self.features_daily_prices = daily_prices
        self.daily_prices = daily_prices[DAY_SHIFT:]
        # # 计算特征值用到的15min数据
        # self.high_freq_data = high_freq_data
        # 交易日数量
        self.days_num = len(self.daily_prices.major_axis)
        # 特征值
        self.features = self.get_features()

        self.quote = Quotes(daily_prices, trading_info)
        self.fac = Market(self.features)
        self.step_counter = 0

    def reset(self):
        self.quote.reset()
        self.step_counter = 0
        return self.fac.step(0)

    def step(self, actions):
        reward, done = self.quote.step(self.step_counter, actions)
        # reward *= 1e2  # 百分之一 基点
        self.step_counter += 1
        next_state = self.fac.step(self.step_counter)
        return next_state, reward, done

    def plot_data(self):
        value = self.quote.buffer_value
        reward = self.quote.buffer_reward
        return value, reward

    def get_features(self):

        dp = self.features_daily_prices
        return get_factor_array(dp, rolling=188)

