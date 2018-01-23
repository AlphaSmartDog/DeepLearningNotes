"""
@author: Young
@license: (C) Copyright 2013-2017
@contact: aidabloc@163.com
@file: train.py
@time: 2018/1/18 22:40
"""
import gc
from itertools import count
import time
import numpy as np
import gym
from agent.agent import Agent
from visdom import Visdom
from config import MEMORY_SIZE

viz = Visdom()
assert viz.check_connection()


MAX_EPISODES = 5000

env = gym.make('BipedalWalker-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = Agent(state_size, action_size, MEMORY_SIZE)

state = env.reset()
for _ in range(MEMORY_SIZE):
    action = agent.get_exploration_policy(state)
    next_state, reward, done, info = env.step(action)
    agent.append(state, action, reward, done, next_state)
    state = next_state
    if done:
        state = env.reset()


def to_np(scale):
    return np.array([scale])


viz_reward = viz.line(X=to_np(0), Y=to_np(0))
time.sleep(1)
viz_length = viz.line(X=to_np(0), Y=to_np(0))

for _ep in range(MAX_EPISODES):
    episode_length = 0
    episode_reward = 0
    state = env.reset()
    for step in count(1):
        # env.render()
        action = agent.get_exploration_policy(state)
        next_state, reward, done, info = env.step(action)
        agent.append(state, action, reward, done, next_state)
        state = next_state
        agent.optimize()

        episode_reward += reward
        if step >= 3000:
            viz.line(X=to_np(_ep+1), Y=to_np(episode_reward), win=viz_reward, update="append")
            time.sleep(0.01)
            viz.line(X=to_np(_ep+1), Y=to_np(step), win=viz_length, update="append")
            break

        if done:
            viz.line(X=to_np(_ep+1), Y=to_np(episode_reward), win=viz_reward, update="append")
            time.sleep(0.01)
            viz.line(X=to_np(_ep+1), Y=to_np(step), win=viz_length, update="append")
            break
    gc.collect()

    if _ep % 1000 == 0:
        print("{} - score: {}".format(_ep, step))
        agent.save_models(_ep)
