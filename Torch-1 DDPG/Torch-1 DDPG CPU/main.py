"""@author: Young
@license: (C) Copyright 2013-2017
@contact: aidabloc@163.com
@file: main.py
@time: 2018/1/17 10:02
"""
import gc
import gym
from agent.agent import Agent


MAX_EPISODES = 5000


env = gym.make('BipedalWalker-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

agent = Agent(state_size, action_size)
state = env.reset()
for _ in range(int(1e3)):
    action = agent.get_exploration_policy(state)
    next_state, reward, done, info = env.step(action)
    agent.append(state, action, reward, done, next_state)
    state = next_state
    if done:
        state = env.reset()


for _ep in range(MAX_EPISODES):
    state = env.reset()
    count = 0
    while True:
        count += 1
        # env.render()
        action = agent.get_exploration_policy(state)
        next_state, reward, done, info = env.step(action)
        agent.append(state, action, reward, done, next_state)
        state = next_state
        agent.optimize()
        if done:
            state = env.reset()
            break
    gc.collect()
    if _ep % 100 == 0:
        print("{} - score: {}".format(_ep, count))
        agent.save_models(_ep)


