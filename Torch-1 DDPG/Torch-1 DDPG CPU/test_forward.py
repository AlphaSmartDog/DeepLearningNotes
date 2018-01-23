#!/usr/bin/env python
# encoding: utf-8
"""
@author: Young
@license: (C) Copyright 2013-2017
@contact: aidabloc@163.com
@file: test_forward.py
@time: 2018/1/17 1:03
"""
import numpy as np
import  torch
from torch.autograd import  Variable
from agent.forward import fan_in_init
from agent.forward import ActorNet, CriticNet

# size = [128, 24]
# # print(fan_in_init(size))
#
# torch_size = torch.Size([128, 24])
# print(fan_in_init(torch_size))

state_size = 24
action_size = 4
np_state = np.float32(np.random.uniform(size=state_size))
np_action = np.float32(np.random.uniform(size=action_size))
A = ActorNet(state_size, action_size)
print(A(Variable(torch.from_numpy(np_state))))


C = CriticNet(state_size, action_size)
# print(C(np_state, np_action))
# print(C(np_state, np_action).shape)
# print(C(np_state, np_action)[0])
# print(torch.squeeze(C(np_state, np_action)))
print(C(Variable(torch.from_numpy (np_state)),
        Variable(torch.from_numpy(np_action))))
