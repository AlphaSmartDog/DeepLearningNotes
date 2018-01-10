import pandas as pd
import numpy as np

DAY_SHIFT = 60

def get_factor_array(dp, rolling=26, drop=False, normalization=True):
    # 各类数据分别装到变量里
    turnovers = dp.total_turnover
    open_interest = dp.open_interest
    opens = dp.open
    closes = dp.close
    lows = dp.low
    highs = dp.high
    stock_list = list(dp.minor_axis)

    # 计算factors - 初始化
    factors = {}
    indexes = closes.index

    # 计算特征值
    for s in stock_list:

        o = opens.loc[:, s]
        c = closes.loc[:, s]
        h = highs.loc[:, s]
        l = lows.loc[:, s]
        v = turnovers.loc[:, s]
        open_int = open_interest.loc[:, s]

        raw_factor = get_factor(indexes.value,
                                o.values,
                                c.values,
                                h.values,
                                l.values,
                                open_int.values,
                                np.array(v, dtype=np.float64),
                                rolling,
                                drop,
                                normalization)
        raw_factor.replace([np.inf, -np.inf], np.nan, inplace=True)
        raw_factor.fillna(0, inplace=True)

        factors[s] = raw_factor

    # 逐日整理
    fac_array = []
    feature_days_num = len(dp.major_axis)
    for i in range(feature_days_num - DAY_SHIFT):
        fac = []
        for s in stock_list:
            real_factor = factors[s].iloc[i: i + DAY_SHIFT]
            fac.append(real_factor)
        fac = np.stack(fac, axis=0)
        fac = np.transpose(fac, [1, 0, 2])
        fac_array.append(fac)

    return fac_array

def get_factor(index,
               opening,
               closing,
               highest,
               lowest,
               volume,
               open_int,
               rolling=26,
               drop=False,
               normalization=True):
    tmp = pd.DataFrame()
    tmp['tradeTime'] = index

    tmp['open'] = opening
    tmp['close'] = closing
    tmp['high'] = highest
    tmp['low'] = lowest
    tmp['volume'] = volume
    tmp['open_int'] = open_int

    # normalize
    if normalization:
        factors_list = tmp.columns.tolist()[1:]

        if rolling >= 26:
            for i in factors_list:
                tmp[i] = (tmp[i] - tmp[i].rolling(window=rolling, center=False).mean()) \
                         / tmp[i].rolling(window=rolling, center=False).std()
        elif rolling < 26 & rolling > 0:
            print('Recommended rolling range greater than 26')
        elif rolling <= 0:
            for i in factors_list:
                tmp[i] = (tmp[i] - tmp[i].mean()) / tmp[i].std()

    if drop:
        tmp.dropna(inplace=True)

    return tmp.set_index('tradeTime')

# def get_factor(index,
#                 opening,
#                 closing,
#                 highest,
#                 lowest,
#                 volume,
#                 open_int,
#                 rolling=26,
#                 drop=False,
#                 normalization=True):
#     tmp = pd.DataFrame()
#     tmp['tradeTime'] = index
#
#
#     # 累积/派发线（Accumulation / Distribution Line，该指标将每日的成交量通过价格加权累计，
#     # 用以计算成交量的动量。属于趋势型因子
#     tmp['AD'] = talib.AD(highest, lowest, closing, volume)
#
#     # 平均动向指数，DMI因子的构成部分。属于趋势型因子
#     tmp['ADX'] = talib.ADX(highest, lowest, closing, timeperiod=14)
#
#     # 相对平均动向指数，DMI因子的构成部分。属于趋势型因子
#     tmp['ADXR'] = talib.ADXR(highest, lowest, closing, timeperiod=14)
#
#     # 均幅指标（Average TRUE Ranger），取一定时间周期内的股价波动幅度的移动平均值，
#     # 是显示市场变化率的指标，主要用于研判买卖时机。属于超买超卖型因子。
#     tmp['ATR14'] = talib.ATR(highest, lowest, closing, timeperiod=14)
#     tmp['ATR6'] = talib.ATR(highest, lowest, closing, timeperiod=6)
#
#     # normalize
#     if normalization:
#         factors_list = tmp.columns.tolist()[1:]
#
#         if rolling >= 26:
#             for i in factors_list:
#                 tmp[i] = (tmp[i] - tmp[i].rolling(window=rolling, center=False).mean()) \
#                          / tmp[i].rolling(window=rolling, center=False).std()
#         elif rolling < 26 & rolling > 0:
#             print('Recommended rolling range greater than 26')
#         elif rolling <= 0:
#             for i in factors_list:
#                 tmp[i] = (tmp[i] - tmp[i].mean()) / tmp[i].std()
#
#     if drop:
#         tmp.dropna(inplace=True)
#
#     return tmp.set_index('tradeTime')
