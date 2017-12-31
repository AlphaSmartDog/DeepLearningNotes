import h5py
import pandas as pd


# DataSet for talib factors image
# Time Uinverse Minutes Factors Days
dataset = h5py.File("dataset_factors.h5")
dataset = dataset["talib_factors"]

# 交易日历
tradeDays = pd.read_hdf("tradeDays.h5").iloc[23:]
tradeDays.reset_index(drop=True, inplace=True)


class Terminal(object):
    def __init__(self):
        self.factors = dataset
        self.tradeDays = tradeDays

    def step(self, step):
        return self.factors[step]

    def reset(self):
        return self.factors[0]
