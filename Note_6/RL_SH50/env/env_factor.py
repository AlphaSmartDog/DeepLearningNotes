import pandas as pd
import talib


def get_factors(index,
                opening,
                closing,
                highest,
                lowest,
                volume,
                rolling=26,
                drop=False,
                normalization=True):
    tmp = pd.DataFrame()
    tmp['tradeTime'] = index

    # 累积/派发线（Accumulation / Distribution Line，该指标将每日的成交量通过价格加权累计，
    # 用以计算成交量的动量。属于趋势型因子
    tmp['AD'] = talib.AD(highest, lowest, closing, volume)

    # 佳庆指标（Chaikin Oscillator），该指标基于AD曲线的指数移动均线而计算得到。属于趋势型因子
    tmp['ADOSC'] = talib.ADOSC(highest, lowest, closing, volume, fastperiod=3, slowperiod=10)

    # 平均动向指数，DMI因子的构成部分。属于趋势型因子
    tmp['ADX'] = talib.ADX(highest, lowest, closing, timeperiod=14)

    # 相对平均动向指数，DMI因子的构成部分。属于趋势型因子
    tmp['ADXR'] = talib.ADXR(highest, lowest, closing, timeperiod=14)

    # 绝对价格振荡指数
    tmp['APO'] = talib.APO(closing, fastperiod=12, slowperiod=26)

    # Aroon通过计算自价格达到近期最高值和最低值以来所经过的期间数，
    # 帮助投资者预测证券价格从趋势到区域区域或反转的变化，
    # Aroon指标分为Aroon、AroonUp和AroonDown3个具体指标。属于趋势型因子
    tmp['AROONDown'], tmp['AROONUp'] = talib.AROON(highest, lowest, timeperiod=14)
    tmp['AROONOSC'] = talib.AROONOSC(highest, lowest, timeperiod=14)

    # 均幅指标（Average TRUE Ranger），取一定时间周期内的股价波动幅度的移动平均值，
    # 是显示市场变化率的指标，主要用于研判买卖时机。属于超买超卖型因子。
    tmp['ATR14'] = talib.ATR(highest, lowest, closing, timeperiod=14)
    tmp['ATR6'] = talib.ATR(highest, lowest, closing, timeperiod=6)

    # 布林带
    tmp['Boll_Up'], tmp['Boll_Mid'], tmp['Boll_Down'] = \
        talib.BBANDS(closing, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # 均势指标
    tmp['BOP'] = talib.BOP(opening, highest, lowest, closing)

    # 5日顺势指标（Commodity Channel Index），专门测量股价是否已超出常态分布范围。属于超买超卖型因子。
    tmp['CCI5'] = talib.CCI(highest, lowest, closing, timeperiod=5)
    tmp['CCI10'] = talib.CCI(highest, lowest, closing, timeperiod=10)
    tmp['CCI20'] = talib.CCI(highest, lowest, closing, timeperiod=20)
    tmp['CCI88'] = talib.CCI(highest, lowest, closing, timeperiod=88)

    # 钱德动量摆动指标（Chande Momentum Osciliator），与其他动量指标摆动指标如
    # 相对强弱指标（RSI）和随机指标（KDJ）不同，
    # 钱德动量指标在计算公式的分子中采用上涨日和下跌日的数据。属于超买超卖型因子
    tmp['CMO_Close'] = talib.CMO(closing, timeperiod=14)
    tmp['CMO_Open'] = talib.CMO(closing, timeperiod=14)

    # DEMA双指数移动平均线
    tmp['DEMA6'] = talib.DEMA(closing, timeperiod=6)
    tmp['DEMA12'] = talib.DEMA(closing, timeperiod=12)
    tmp['DEMA26'] = talib.DEMA(closing, timeperiod=26)

    # DX 动向指数
    tmp['DX'] = talib.DX(highest, lowest, closing, timeperiod=14)

    # EMA 指数移动平均线
    tmp['EMA6'] = talib.EMA(closing, timeperiod=6)
    tmp['EMA12'] = talib.EMA(closing, timeperiod=12)
    tmp['EMA26'] = talib.EMA(closing, timeperiod=26)

    # KAMA 适应性移动平均线
    tmp['KAMA'] = talib.KAMA(closing, timeperiod=30)

    # MACD
    tmp['MACD_DIF'], tmp['MACD_DEA'], tmp['MACD_bar'] = \
        talib.MACD(closing, fastperiod=12, slowperiod=24, signalperiod=9)

    # 中位数价格 不知道是什么意思
    tmp['MEDPRICE'] = talib.MEDPRICE(highest, lowest)

    # 负向指标 负向运动
    tmp['MiNUS_DI'] = talib.MINUS_DI(highest, lowest, closing, timeperiod=14)
    tmp['MiNUS_DM'] = talib.MINUS_DM(highest, lowest, timeperiod=14)

    # 动量指标（Momentom Index），动量指数以分析股价波动的速度为目的，研究股价在波动过程中各种加速，
    # 减速，惯性作用以及股价由静到动或由动转静的现象。属于趋势型因子
    tmp['MOM'] = talib.MOM(closing, timeperiod=10)

    # 归一化平均值范围
    tmp['NATR'] = talib.NATR(highest, lowest, closing, timeperiod=14)

    # OBV 	能量潮指标（On Balance Volume，OBV），以股市的成交量变化来衡量股市的推动力，
    # 从而研判股价的走势。属于成交量型因子
    tmp['OBV'] = talib.OBV(closing, volume)

    # PLUS_DI 更向指示器
    tmp['PLUS_DI'] = talib.PLUS_DI(highest, lowest, closing, timeperiod=14)
    tmp['PLUS_DM'] = talib.PLUS_DM(highest, lowest, timeperiod=14)

    # PPO 价格振荡百分比
    tmp['PPO'] = talib.PPO(closing, fastperiod=6, slowperiod=26, matype=0)

    # ROC 6日变动速率（Price Rate of Change），以当日的收盘价和N天前的收盘价比较，
    # 通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量。属于超买超卖型因子。
    tmp['ROC6'] = talib.ROC(closing, timeperiod=6)
    tmp['ROC20'] = talib.ROC(closing, timeperiod=20)
    # 12日量变动速率指标（Volume Rate of Change），以今天的成交量和N天前的成交量比较，
    # 通过计算某一段时间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，
    # 达到事先探测成交量供需的强弱，进而分析成交量的发展趋势及其将来是否有转势的意愿，
    # 属于成交量的反趋向指标。属于成交量型因子
    tmp['VROC6'] = talib.ROC(volume, timeperiod=6)
    tmp['VROC20'] = talib.ROC(volume, timeperiod=20)

    # ROC 6日变动速率（Price Rate of Change），以当日的收盘价和N天前的收盘价比较，
    # 通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量。属于超买超卖型因子。
    tmp['ROCP6'] = talib.ROCP(closing, timeperiod=6)
    tmp['ROCP20'] = talib.ROCP(closing, timeperiod=20)
    # 12日量变动速率指标（Volume Rate of Change），以今天的成交量和N天前的成交量比较，
    # 通过计算某一段时间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，
    # 达到事先探测成交量供需的强弱，进而分析成交量的发展趋势及其将来是否有转势的意愿，
    # 属于成交量的反趋向指标。属于成交量型因子
    tmp['VROCP6'] = talib.ROCP(volume, timeperiod=6)
    tmp['VROCP20'] = talib.ROCP(volume, timeperiod=20)

    # RSI
    tmp['RSI'] = talib.RSI(closing, timeperiod=14)

    # SAR 抛物线转向
    tmp['SAR'] = talib.SAR(highest, lowest, acceleration=0.02, maximum=0.2)

    # TEMA
    tmp['TEMA6'] = talib.TEMA(closing, timeperiod=6)
    tmp['TEMA12'] = talib.TEMA(closing, timeperiod=12)
    tmp['TEMA26'] = talib.TEMA(closing, timeperiod=26)

    # TRANGE 真实范围
    tmp['TRANGE'] = talib.TRANGE(highest, lowest, closing)

    # TYPPRICE 典型价格
    tmp['TYPPRICE'] = talib.TYPPRICE(highest, lowest, closing)

    # TSF 时间序列预测
    tmp['TSF'] = talib.TSF(closing, timeperiod=14)

    # ULTOSC 极限振子
    tmp['ULTOSC'] = talib.ULTOSC(highest, lowest, closing, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # 威廉指标
    tmp['WILLR'] = talib.WILLR(highest, lowest, closing, timeperiod=14)

    # 标准化
    if normalization:
        factors_list = tmp.columns.tolist()[1:]

        if rolling >= 26:
            for i in factors_list:
                tmp[i] = (tmp[i] - tmp[i].rolling(window=rolling, center=False).mean())\
                         / tmp[i].rolling(window=rolling, center=False).std()
        elif rolling < 26 & rolling > 0:
            print('Recommended rolling range greater than 26')
        elif rolling <= 0:
            for i in factors_list:
                tmp[i] = (tmp[i] - tmp[i].mean()) / tmp[i].std()

    if drop:
        tmp.dropna(inplace=True)

    return tmp.set_index('tradeTime')