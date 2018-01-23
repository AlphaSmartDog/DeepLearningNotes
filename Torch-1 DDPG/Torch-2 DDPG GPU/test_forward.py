"""
@author: Young
@license: (C) Copyright 2013-2017
@contact: aidabloc@163.com
@file: test_forward.py
@time: 2018/1/17 22:01
"""
import numpy as np
import torch
from torch.autograd import Variable
from agent.forward import ActorNet, CriticNet
import gym


env = gym.make('BipedalWalker-v2')
state = env.reset()
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

A = ActorNet(state_size, action_size)
C = CriticNet(state_size, action_size)

print(state.dtype)
state = Variable(torch.from_numpy(np.float32(state)))
print(A(state))
print(C(state, A(state)))