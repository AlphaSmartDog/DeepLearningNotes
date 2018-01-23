"""
@author: Young
@license: (C) Copyright 2013-2017
@contact: aidabloc@163.com
@file: noise.py
@time: 2018/1/18 21:47
"""
import numpy as np


class Noise(object):
    def __init__(self, action_size, mu=0, theta=0.15, sigma=0.2):
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_size) * self.mu

    def reset(self):
        self.X = np.ones(self.action_size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X += dx
        return self.X

    def __call__(self):
        return self.sample()