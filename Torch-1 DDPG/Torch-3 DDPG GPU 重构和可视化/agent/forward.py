"""
@author: Young
@license: (C) Copyright 2013-2017
@contact: aidabloc@163.com
@file: forward.py
@time: 2018/1/18 20:16
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nf


def _swich(tensor):
    return tensor * nf.sigmoid(tensor)


def _init_fan_in(data):
    size = data.size()
    value = 1. / np.sqrt(size[0])
    return torch.FloatTensor(size).uniform_(-value, value)


class ActorNet(nn.Module):
    def __init__(self, state_size, action_size, init_w=3e-3):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc1.weight.data = _init_fan_in(self.fc1.weight.data)
        self.fc2 = nn.Linear(64, 64)
        self.fc2.weight.data = _init_fan_in(self.fc2.weight.data)
        self.fc3 = nn.Linear(64, action_size)
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        net = nf.relu(self.fc1(state))
        net = nf.relu(self.fc2(net))
        return nf.tanh(self.fc3(net))


class CriticNet(nn.Module):
    def __init__(self, state_size, action_size, init_w=3e-3):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc1.weight.data = _init_fan_in(self.fc1.weight.data)
        self.fc2 = nn.Linear(64+action_size, 64)
        self.fc2.weight.data = _init_fan_in(self.fc2.weight.data)
        self.fc3 = nn.Linear(64, 1)
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        net = nf.relu(self.fc1(state))
        net = torch.cat((net, action), -1)
        net = nf.relu(self.fc2(net))
        return self.fc3(net)
