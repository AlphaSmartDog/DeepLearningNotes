"""
@author: Young
@license: (C) Copyright 2013-2017
@contact: aidabloc@163.com
@file: access.py
@time: 2018/1/18 20:51
"""
from collections import namedtuple
import random
import numpy as np


Memory = namedtuple("Memory",
                    ["state", "action", "reward", "done", "next_state"])


class Access(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = [None] * capacity
        self.pointer = 0

    def __len__(self):
        return self.capacity

    def __call__(self, *args, **kwargs):
        self.append(*args)

    def append(self, *args):
        self.cache[self.pointer] = Memory(*args)
        self.pointer = (self.pointer + 1) % self.capacity

    def sample(self, batch_size=64):
        buffer = zip(*random.sample(self.cache, batch_size))
        return [np.array(i) for i in buffer]






