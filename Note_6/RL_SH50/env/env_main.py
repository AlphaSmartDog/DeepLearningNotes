from env.env_market import Market
from env.env_quotes import Quotes
from env.env_factor import get_factors
import numpy as np
import pandas as pd


class Account(object):
    def __init__(self, daily_prices, high_freq_data):
        # 日线数据
        self.daily_prices = daily_prices
        # 计算特征值用到的15min数据
        self.high_freq_data = high_freq_data
        # 交易日数量
        self.days_num = len(self.daily_prices.major_axis)
        # 特征值
        self.features = self.get_features()

        self.quote = Quotes(daily_prices)
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

        dp = self.high_freq_data
        # 获取Quotes的第一天日期
        target_date = self.daily_prices.major_axis[0]

        # 各类数据分别装到变量里
        turnovers = dp.total_turnover
        opens = dp.open
        closes = dp.close
        lows = dp.low
        highs = dp.high
        universe = list(dp.minor_axis)

        # 计算factors - 初始化
        factors = {}
        indexes = closes.index

        # 计算特征值
        for i in universe:

            o = opens.loc[:, i]
            c = closes.loc[:, i]
            h = highs.loc[:, i]
            l = lows.loc[:, i]
            v = turnovers.loc[:, i]
            tmp = get_factors(indexes.values,
                              o.values,
                              c.values,
                              h.values,
                              l.values,
                              np.array(v, dtype=np.float64),
                              rolling=188,
                              drop=False)
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            tmp.fillna(0, inplace=True)

            # 找到Quotes第一天的开始
            for d in tmp.index:
                if d.year == target_date.year and d.month == target_date.month and d.day == target_date.day:
                    my_day = d
                    break
            location_of_my_day = tmp.index.get_loc(my_day)
            # 训练要取前几天的特征值，因此start_location要做对应调整
            start_location = location_of_my_day - 64

            factors[i] = tmp.iloc[start_location:]

        # 逐日整理
        fac_array = []
        for i in range(self.days_num):
            j = i * 16 + 64
            fac = []
            for k in universe:
                tmp = factors[k]
                tmp = tmp.iloc[j - 16 * 4: j]
                fac.append(tmp)
            fac = np.stack(fac, axis=0)
            fac = np.transpose(fac, [1, 0, 2])
            fac_array.append(fac)

        return (fac_array)
