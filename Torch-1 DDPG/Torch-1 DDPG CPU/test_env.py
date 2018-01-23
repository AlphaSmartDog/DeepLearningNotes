#!/usr/bin/env python
# encoding: utf-8
"""
@author: Young
@license: (C) Copyright 2013-2017
@contact: aidabloc@163.com
@file: test_env.py
@time: 2018/1/16 23:41
"""

import gym

env = gym.make('BipedalWalker-v2')

print(env.reset())
print(env.action_space)
print(env.observation_space)

print(env.action_space.sample())
print(env.action_space.high)
print(env.action_space.low)

state = env.reset()
print(state.shape)
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)
print(next_state.shape)
print(reward)
print(done)
print(info)

print(env.reward_range)

while True:
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print(reward)
    if done:
        print(reward)
        break

# state 24
# action  4
# reward -1 ~ 1  done reward -100

