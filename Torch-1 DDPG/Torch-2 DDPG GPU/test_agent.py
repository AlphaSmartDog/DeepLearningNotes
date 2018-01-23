"""
@author: Young
@license: (C) Copyright 2013-2017
@contact: aidabloc@163.com
@file: main.py
@time: 2018/1/17 22:25
"""
import os
import gc
import gym
import numpy as np
import torch
from torch.autograd import Variable
from agent.agent import Agent


MAX_EPISODES = 5000


env = gym.make('BipedalWalker-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

agent = Agent(state_size, action_size)
state = env.reset()
agent.get_exploration_policy(state)

for _ in range(int(1e3)):
    action = agent.get_exploration_policy(state)
    next_state, reward, done, info = env.step(action)
    agent.append(state, action, reward, done, next_state)
    state = next_state
    if done:
        state = env.reset()

agent.optimize()

agent.save_models(1)




