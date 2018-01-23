"""@author: Young
@license: (C) Copyright 2013-2017
@contact: aidabloc@163.com
@file: test_agent.py
@time: 2018/1/17 10:19
"""
import gc
import gym
import numpy as np

from agent.agent import Agent

env = gym.make('BipedalWalker-v2')
state = env.reset()
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
action_limits = env.action_space.high[0]

agent = Agent(state_size, action_size, action_limits)
for i in range(128):
    action = agent.get_exploration_policy(state)
    next_state, reward, done, info = env.step(action)
    agent.append(state, action, reward, done, next_state)

agent.optimize()
print("end")
agent.optimize()