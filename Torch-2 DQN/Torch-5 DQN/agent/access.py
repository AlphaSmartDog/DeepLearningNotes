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
        self.length = 0

    def reset(self):
        self.cache = [None] * self.capacity
        self.pointer = 0
        self.length = 0

    def __len__(self):
        return self.capacity

    def __call__(self, *args, **kwargs):
        self.append(*args)

    def append(self, *args):
        self.cache[self.pointer] = Memory(*args)
        self.pointer = (self.pointer + 1) % self.capacity
        if self.length < self.capacity:
            self.length += 1
        elif self.length >= self.capacity:
            pass

    def sample(self, batch_size=64):
        if self.length < self.capacity:
            buffer = zip(*random.sample(
                self.cache[:self.length], batch_size))
        else:
            buffer = zip(*random.sample(self.cache, batch_size))
        return [np.array(i) for i in buffer]
