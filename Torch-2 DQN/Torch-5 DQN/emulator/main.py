import numpy as np
import pandas as pd
from emulator.env_data import high2low, fix_data
from emulator.env_factor import get_factors

quotes = fix_data('emulator/HS300.csv')
quotes = high2low(quotes, '5min')
daily_quotes = high2low(quotes, '1d')

Index = quotes.index
High = quotes.high.values
Low = quotes.low.values
Close = quotes.close.values
Open = quotes.open.values
Volume = quotes.volume.values
factors = get_factors(Index, Open, Close, High, Low, Volume, rolling=188, drop=True)

daily_quotes['returns'] = np.log(daily_quotes['close'].shift(-1) / daily_quotes['close'])
daily_quotes.dropna(inplace=True)

start_date = pd.to_datetime('2011-01-12')
end_date = pd.to_datetime('2016-12-29')
daily_quotes = daily_quotes.loc[start_date:end_date]
daily_quotes = daily_quotes.iloc[5:]
factors = factors.loc[start_date:end_date]

fac_list = []
for i in range(len(daily_quotes)):
    s = i * 50
    e = (i + 5) * 50
    f = np.array(factors.iloc[s:e])
    fac_list.append(np.expand_dims(f, axis=0))

fac_array = np.concatenate(fac_list, axis=0)
shape = [fac_array.shape[0], 5, 50, fac_array.shape[2]]
fac_array = fac_array.reshape(shape)
fac_array = np.transpose(fac_array, [0, 2, 3, 1])

DATE_QUOTES = daily_quotes
DATA_FAC = fac_array


class Account(object):
    def __init__(self):
        self.data_close = DATE_QUOTES['close']
        self.data_open = DATE_QUOTES['open']
        self.data_observation = DATA_FAC
        self.action_space = ['long', 'short', 'close']
        self.free = 1e-4
        self.step_counter = 0
        self.cash = 1e5
        self.position = 0
        self.total_value = self.cash + self.position
        self.flags = 0

    def reset(self):
        self.step_counter = 0
        self.cash = 1e5
        self.position = 0
        self.total_value = self.cash + self.position
        self.flags = 0
        return self._get_initial_state()

    def _get_initial_state(self):
        return np.transpose(self.data_observation[0], [2, 0, 1])

    def get_action_space(self):
        return self.action_space

    def long(self):
        self.flags = 1
        quotes = self.data_open[self.step_counter] * 10
        self.cash -= quotes * (1 + self.free)
        self.position = quotes

    def short(self):
        self.flags = -1
        quotes = self.data_open[self.step_counter] * 10
        self.cash += quotes * (1 - self.free)
        self.position = - quotes

    def keep(self):
        quotes = self.data_open[self.step_counter] * 10
        self.position = quotes * self.flags

    def close_long(self):
        self.flags = 0
        quotes = self.data_open[self.step_counter] * 10
        self.cash += quotes * (1 - self.free)
        self.position = 0

    def close_short(self):
        self.flags = 0
        quotes = self.data_open[self.step_counter] * 10
        self.cash -= quotes * (1 + self.free)
        self.position = 0

    def step_op(self, action):

        if action == 'long':
            if self.flags == 0:
                self.long()
            elif self.flags == -1:
                self.close_short()
                self.long()
            else:
                self.keep()

        elif action == 'close':
            if self.flags == 1:
                self.close_long()
            elif self.flags == -1:
                self.close_short()
            else:
                pass

        elif action == 'short':
            if self.flags == 0:
                self.short()
            elif self.flags == 1:
                self.close_long()
                self.short()
            else:
                self.keep()
        else:
            raise ValueError("action should be elements of ['long', 'short', 'close']")

        position = self.data_close[self.step_counter] * 10 * self.flags
        reward = np.log((self.cash + position)/self.total_value)
        self.step_counter += 1
        self.total_value = position + self.cash
        next_observation = self.data_observation[self.step_counter]
        done = False
        if self.total_value < 4000:
            done = True
        if self.step_counter > 1000:
            done = True
        return np.transpose(next_observation, [2, 0, 1]), reward, done

    def step(self, action):
        if action == 0:
            return self.step_op('long')
        elif action == 1:
            return self.step_op('short')
        elif action == 2:
            return self.step_op('close')
        else:
            raise ValueError("action should be one of [0,1,2]")









