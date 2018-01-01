import numpy as np
import pandas as pd
from emulator.env_factor import get_factors

turnovers = pd.read_excel('emulator/SH50.xls', sheetname='total_turnover')
opens = pd.read_excel('emulator/SH50.xls', sheetname='open')
closes = pd.read_excel('emulator/SH50.xls', sheetname='close')
highs = pd.read_excel('emulator/SH50.xls', sheetname='high')
lows = pd.read_excel('emulator/SH50.xls', sheetname='low')


universe = ['600048.XSHG', '601601.XSHG', '600887.XSHG',
            '600109.XSHG', '601186.XSHG', '600030.XSHG',
            '601169.XSHG', '601398.XSHG', '601088.XSHG',
            '600028.XSHG', '601336.XSHG', '601901.XSHG',
            '600050.XSHG', '600000.XSHG', '600485.XSHG',
            '601288.XSHG', '601377.XSHG', '600029.XSHG',
            '601198.XSHG', '601211.XSHG', '600893.XSHG',
            '600547.XSHG', '601668.XSHG', '601166.XSHG',
            '600100.XSHG', '601006.XSHG', '601818.XSHG',
            '600036.XSHG', '600837.XSHG', '600958.XSHG',
            '601318.XSHG', '600016.XSHG', '601628.XSHG',
            '601800.XSHG', '601688.XSHG', '601857.XSHG',
            '601989.XSHG', '601998.XSHG', '600999.XSHG',
            '600518.XSHG', '601328.XSHG', '601988.XSHG',
            '601788.XSHG', '600637.XSHG', '600104.XSHG',
            '601390.XSHG', '600111.XSHG', '601766.XSHG',
            '600519.XSHG', '601985.XSHG']
# universe.sort()

factors = {}
indexes = closes['index']
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
    factors[i] = tmp

fac_array = []
for i in range(202):
    j = i * 16 + 351
    fac = []
    for k in universe:
        tmp = factors[k]
        tmp = tmp.iloc[j-16*4: j]
        fac.append(tmp)
    fac = np.stack(fac, axis=0)
    fac = np.transpose(fac, [1, 0, 2])
    fac_array.append(fac)


# class Market(object):
#     def __init__(self):
#         self.fac = fac_array
#
#     def step(self, step_counter):
#         return self.fac[step_counter]

class Market(object):
    def __init__(self):
        pass

    def step(self, step_counter):
        return fac_array[step_counter]

