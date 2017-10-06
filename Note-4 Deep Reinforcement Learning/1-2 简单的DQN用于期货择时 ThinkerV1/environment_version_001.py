
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import talib 

def fix_data(path):
    tmp = pd.read_csv(path, encoding="gbk", engine='python')
    tmp.rename(columns={'Unnamed: 0':'trading_time'}, inplace=True)
    tmp['trading_point'] = pd.to_datetime(tmp.trading_time)
    del tmp['trading_time']
    tmp.set_index(tmp.trading_point, inplace=True)
    return tmp

def High_2_Low(tmp, freq):
    """处理从RiceQuant下载的分钟线数据，
    从分钟线数据合成低频数据
    2017-08-11    
    """
    # 分别处理bar数据
    tmp_open = tmp['open'].resample(freq).ohlc()
    tmp_open = tmp_open['open'].dropna()

    tmp_high = tmp['high'].resample(freq).ohlc()
    tmp_high = tmp_high['high'].dropna()

    tmp_low = tmp['low'].resample(freq).ohlc()
    tmp_low = tmp_low['low'].dropna()

    tmp_close = tmp['close'].resample(freq).ohlc()
    tmp_close = tmp_close['close'].dropna()

    tmp_price = pd.concat([tmp_open, tmp_high, tmp_low, tmp_close], axis=1)
    
    # 处理成交量
    tmp_volume = tmp['volume'].resample(freq).sum()
    tmp_volume.dropna(inplace=True)
    
    return pd.concat([tmp_price, tmp_volume], axis=1)

def get_factors(index, 
                Open, 
                Close, 
                High, 
                Low, 
                Volume,
                rolling = 26,
                drop=False, 
                normalization=True):
    
    tmp = pd.DataFrame()
    tmp['tradeTime'] = index
    
    #累积/派发线（Accumulation / Distribution Line，该指标将每日的成交量通过价格加权累计，
    #用以计算成交量的动量。属于趋势型因子
    tmp['AD'] = talib.AD(High, Low, Close, Volume)

    # 佳庆指标（Chaikin Oscillator），该指标基于AD曲线的指数移动均线而计算得到。属于趋势型因子
    tmp['ADOSC'] = talib.ADOSC(High, Low, Close, Volume, fastperiod=3, slowperiod=10)

    # 平均动向指数，DMI因子的构成部分。属于趋势型因子
    tmp['ADX'] = talib.ADX(High, Low, Close,timeperiod=14)

    # 相对平均动向指数，DMI因子的构成部分。属于趋势型因子
    tmp['ADXR'] = talib.ADXR(High, Low, Close,timeperiod=14)

    # 绝对价格振荡指数
    tmp['APO'] = talib.APO(Close, fastperiod=12, slowperiod=26)

    # Aroon通过计算自价格达到近期最高值和最低值以来所经过的期间数，帮助投资者预测证券价格从趋势到区域区域或反转的变化，
    #Aroon指标分为Aroon、AroonUp和AroonDown3个具体指标。属于趋势型因子
    tmp['AROONDown'], tmp['AROONUp'] = talib.AROON(High, Low,timeperiod=14)
    tmp['AROONOSC'] = talib.AROONOSC(High, Low,timeperiod=14)

    # 均幅指标（Average TRUE Ranger），取一定时间周期内的股价波动幅度的移动平均值，
    #是显示市场变化率的指标，主要用于研判买卖时机。属于超买超卖型因子。
    tmp['ATR14']= talib.ATR(High, Low, Close, timeperiod=14)
    tmp['ATR6']= talib.ATR(High, Low, Close, timeperiod=6)

    # 布林带
    tmp['Boll_Up'],tmp['Boll_Mid'],tmp['Boll_Down']= talib.BBANDS(Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # 均势指标
    tmp['BOP'] = talib.BOP(Open, High, Low, Close)

    #5日顺势指标（Commodity Channel Index），专门测量股价是否已超出常态分布范围。属于超买超卖型因子。
    tmp['CCI5'] = talib.CCI(High, Low, Close, timeperiod=5)
    tmp['CCI10'] = talib.CCI(High, Low, Close, timeperiod=10)
    tmp['CCI20'] = talib.CCI(High, Low, Close, timeperiod=20)
    tmp['CCI88'] = talib.CCI(High, Low, Close, timeperiod=88)

    # 钱德动量摆动指标（Chande Momentum Osciliator），与其他动量指标摆动指标如相对强弱指标（RSI）和随机指标（KDJ）不同，
    # 钱德动量指标在计算公式的分子中采用上涨日和下跌日的数据。属于超买超卖型因子
    tmp['CMO_Close'] = talib.CMO(Close,timeperiod=14)
    tmp['CMO_Open'] = talib.CMO(Close,timeperiod=14)

    # DEMA双指数移动平均线
    tmp['DEMA6'] = talib.DEMA(Close, timeperiod=6)
    tmp['DEMA12'] = talib.DEMA(Close, timeperiod=12)
    tmp['DEMA26'] = talib.DEMA(Close, timeperiod=26)

    # DX 动向指数
    tmp['DX'] = talib.DX(High, Low, Close,timeperiod=14)

    # EMA 指数移动平均线
    tmp['EMA6'] = talib.EMA(Close, timeperiod=6)
    tmp['EMA12'] = talib.EMA(Close, timeperiod=12)
    tmp['EMA26'] = talib.EMA(Close, timeperiod=26)

    # KAMA 适应性移动平均线
    tmp['KAMA'] = talib.KAMA(Close, timeperiod=30)

    # MACD
    tmp['MACD_DIF'],tmp['MACD_DEA'],tmp['MACD_bar'] = talib.MACD(Close, fastperiod=12, slowperiod=24, signalperiod=9)

    # 中位数价格 不知道是什么意思
    tmp['MEDPRICE'] = talib.MEDPRICE(High, Low)

    # 负向指标 负向运动
    tmp['MiNUS_DI'] = talib.MINUS_DI(High, Low, Close,timeperiod=14)
    tmp['MiNUS_DM'] = talib.MINUS_DM(High, Low,timeperiod=14)

    # 动量指标（Momentom Index），动量指数以分析股价波动的速度为目的，研究股价在波动过程中各种加速，
    #减速，惯性作用以及股价由静到动或由动转静的现象。属于趋势型因子
    tmp['MOM'] = talib.MOM(Close, timeperiod=10)

    # 归一化平均值范围
    tmp['NATR'] = talib.NATR(High, Low, Close,timeperiod=14)

    # OBV 	能量潮指标（On Balance Volume，OBV），以股市的成交量变化来衡量股市的推动力，
    #从而研判股价的走势。属于成交量型因子
    tmp['OBV'] = talib.OBV(Close, Volume)

    # PLUS_DI 更向指示器
    tmp['PLUS_DI'] = talib.PLUS_DI(High, Low, Close,timeperiod=14)
    tmp['PLUS_DM'] = talib.PLUS_DM(High, Low, timeperiod=14)

    # PPO 价格振荡百分比
    tmp['PPO'] = talib.PPO(Close, fastperiod=6, slowperiod= 26, matype=0)

    # ROC 6日变动速率（Price Rate of Change），以当日的收盘价和N天前的收盘价比较，
    #通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量。属于超买超卖型因子。
    tmp['ROC6'] = talib.ROC(Close, timeperiod=6)
    tmp['ROC20'] = talib.ROC(Close, timeperiod=20)
    #12日量变动速率指标（Volume Rate of Change），以今天的成交量和N天前的成交量比较，
    #通过计算某一段时间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，
    #达到事先探测成交量供需的强弱，进而分析成交量的发展趋势及其将来是否有转势的意愿，
    #属于成交量的反趋向指标。属于成交量型因子
    tmp['VROC6'] = talib.ROC(Volume, timeperiod=6)
    tmp['VROC20'] = talib.ROC(Volume, timeperiod=20)

    # ROC 6日变动速率（Price Rate of Change），以当日的收盘价和N天前的收盘价比较，
    #通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量。属于超买超卖型因子。
    tmp['ROCP6'] = talib.ROCP(Close, timeperiod=6)
    tmp['ROCP20'] = talib.ROCP(Close, timeperiod=20)
    #12日量变动速率指标（Volume Rate of Change），以今天的成交量和N天前的成交量比较，
    #通过计算某一段时间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，
    #达到事先探测成交量供需的强弱，进而分析成交量的发展趋势及其将来是否有转势的意愿，
    #属于成交量的反趋向指标。属于成交量型因子
    tmp['VROCP6'] = talib.ROCP(Volume, timeperiod=6)
    tmp['VROCP20'] = talib.ROCP(Volume, timeperiod=20)

    # RSI
    tmp['RSI'] = talib.RSI(Close, timeperiod=14)

    # SAR 抛物线转向
    tmp['SAR'] = talib.SAR(High, Low, acceleration=0.02, maximum=0.2)

    # TEMA
    tmp['TEMA6'] = talib.TEMA(Close, timeperiod=6)
    tmp['TEMA12'] = talib.TEMA(Close, timeperiod=12)
    tmp['TEMA26'] = talib.TEMA(Close, timeperiod=26)

    # TRANGE 真实范围
    tmp['TRANGE'] = talib.TRANGE(High, Low, Close)

    # TYPPRICE 典型价格
    tmp['TYPPRICE'] = talib.TYPPRICE(High, Low, Close)

    # TSF 时间序列预测
    tmp['TSF'] = talib.TSF(Close, timeperiod=14)

    # ULTOSC 极限振子
    tmp['ULTOSC'] = talib.ULTOSC(High, Low, Close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # 威廉指标
    tmp['WILLR'] = talib.WILLR(High, Low, Close, timeperiod=14)
    
    # 标准化
    if normalization:
        factors_list = tmp.columns.tolist()[1:]

        if rolling >= 26:
            for i in factors_list:
                tmp[i] = (tmp[i] - tmp[i].rolling(window=rolling, center=False).mean())                /tmp[i].rolling(window=rolling, center=False).std()
        elif rolling < 26 & rolling > 0:
            print ('Recommended rolling range greater than 26')
        elif rolling <=0:
            for i in factors_list:
                tmp[i] = (tmp[i] - tmp[i].mean())/tmp[i].std()
            
    if drop:
        tmp.dropna(inplace=True)
        
    tmp.set_index('tradeTime', inplace=True)
    
    return tmp


# In[ ]:

tmp = fix_data('HS300.csv')
tmp = High_2_Low(tmp, '5min')
Index = tmp.index
High = tmp.high.values
Low = tmp.low.values
Close = tmp.close.values
Open = tmp.open.values
Volume = tmp.volume.values
factors = get_factors(Index, Open, Close, High, Low, Volume, rolling = 188, drop=True)
tmp['returns'] = np.log(tmp['close'].shift(-1)/tmp['close'])

data = pd.merge(tmp, factors, left_index=True, right_index=True, how='outer').dropna()
data.reset_index(drop=True, inplace=True)

#rewards_Series = data['returns']
#observations_DataFrame = data.iloc[:,6:]


# In[ ]:

class Position(object):
    def __init__(self, data):
        self.data = data
        self.data_closes = data['close']
        self.data_observations = data.iloc[:,6:]
        self.action_space = ['long', 'short', 'keep', 'close']
        self.cash = 1e4
        self.position = 0
        self.total_value = self.cash + self.position
        self.flags = 0
        self.free = 5e-5
        self.step_counter = 0
    
    def long(self):
        self.flags = 1
        quotes = self.data_closes[self.step_counter]
        self.cash -= quotes * (1 + self.free)
        self.position = quotes
        
    def short(self):
        self.flags = -1
        quotes = self.data_closes[self.step_counter]
        self.cash += quotes * (1 - self.free)
        self.position = - quotes        
        
    def keep(self):
        quotes = self.data_closes[self.step_counter]
        self.position = quotes * self.flags
        
    def close_long(self): 
        self.flags = 0
        quotes = self.data_closes[self.step_counter]
        self.cash += quotes * (1 - self.free)
        self.position = 0  
        
    def close_short(self):
        self.flags = 0
        quotes = self.data_closes[self.step_counter]
        self.cash -= quotes * (1 + self.free)
        self.position = 0  
    
    def reset(self):
        self.step_counter = 0
        self.cash = 1e4
        self.position = 0
        self.total_value = self.cash + self.position
        self.flags = 0
        
    def step(self, action):
        if action == 0:
            return self.step_op('long')
        elif action == 1:
            return self.step_op('short')
        elif action == 2:
            return self.step_op('keep')
        elif action == 3:
            return self.step_op('close')
        else:
            raise ValueError("action should be one of [0,1,2,3]")
        
    def step_op(self, action):
        
        if action == 'long':
            if self.flags == 0:
                self.long()
            elif self.flags == -1:
                self.close_short()
                self.long()
            else:
                self.keep() 
                
        elif action == 'keep':
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
            raise ValueError("action should be elements of ['long', 'short', 'keep', 'close']")
            
        if self.total_value < 4000:
            done = True
        elif self.step_counter > 6000:
            done = True
        else:
            done = False
        
        step_reward = self.position + self.cash - self.total_value
        self.total_value = self.position + self.cash
        self.step_counter += 1
        next_observation = self.data_observations.iloc[self.step_counter]
        return step_reward, next_observation, done
    
    def get_initial_state(self):
        return self.data_observations.iloc[0]


# In[ ]:

env = Position(data)

