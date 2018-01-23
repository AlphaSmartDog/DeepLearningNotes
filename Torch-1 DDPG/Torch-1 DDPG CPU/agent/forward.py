#!/usr/bin/env python
# encoding: utf-8
"""
@author: Young
@license: (C) Copyright 2013-2017
@contact: aidabloc@163.com
@file: forward.py
@time: 2018/1/16 22:34
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as nf

from config import INIT_W


def fan_in_init(size, fan_in=None):
    if not isinstance(size, torch.Size):
        raise ValueError("size should be torch.Size")
    fan_in = fan_in or size[0]
    value = 1. / np.sqrt(fan_in)
    return torch.FloatTensor(size).uniform_(-value, value)


class ActorNet(nn.Module):
    def __init__(self, state_size, action_size,
                 action_norm=1.):
        super().__init__()
        self.action_norm = action_norm
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_size)
        self.init_weights(INIT_W)

    def init_weights(self, init_w):
        self.fc1.weight.data = fan_in_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fan_in_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        net = nf.relu(self.fc1(state))
        net = nf.relu(self.fc2(net))
        return nf.tanh(self.fc3(net)) * self.action_norm


class CriticNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400+action_size, 300)
        self.fc3 = nn.Linear(300, 1)
        self.init_weights(INIT_W)

    def init_weights(self, init_w):
        self.fc1.weight.data = fan_in_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fan_in_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        net = nf.relu(self.fc1(state))
        net = torch.cat((net, action), -1)
        net = nf.relu(self.fc2(net))
        return self.fc3(net)
