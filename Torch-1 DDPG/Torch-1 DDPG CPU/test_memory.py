#!/usr/bin/env python
# encoding: utf-8
"""
@author: Young
@license: (C) Copyright 2013-2017
@contact: aidabloc@163.com
@file: test_memory.py
@time: 2018/1/16 21:37
"""
import numpy as np
from agent.memory import Memory

M = Memory()

state = np.random.normal(size=24)
action = np.random.normal(size=4)
reward = np.random.normal()
done = np.bool(np.random.randint(0, 2))
next_state = state
for _ in range(int(1e6)):
    M(state, action, reward, done, next_state)

states, actions, rewards, next_states = M.sample(128)
print(states.shape)
print(actions.shape)
print(rewards.shape)
print(next_states.shape)