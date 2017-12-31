import h5py
import pandas as pd


# DataSet for talib factors image
# Time Uinverse Minutes Factors Days
dataset = h5py.File("emulator_v0/dataset_factors.h5")
dataset = dataset["talib_factors"]
# 交易日历
tradeDays = pd.read_hdf("emulator_v0/tradeDays.h5").iloc[23:]
tradeDays.reset_index(drop=True, inplace=True)

# 股票池
dataset_universe = pd.read_hdf("emulator_v0/universe_SH50.h5")


class Market(object):
    def __init__(self):
        self.factors = dataset
        self.tradeDays = tradeDays
        self.universe = dataset_universe

    def step(self, counter):
        day = self.tradeDays[counter]
        return self.factors[counter], self.universe.loc[day].tolist()

    def reset(self):
        return self.factors[0], self.universe.iloc[0].tolist()
