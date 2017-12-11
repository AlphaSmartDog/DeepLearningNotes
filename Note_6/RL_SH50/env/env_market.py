import numpy as np
import pandas as pd


class Market(object):
    def __init__(self, features):
        self.fac_array = features

    def step(self, step_counter):
        return self.fac_array[step_counter]