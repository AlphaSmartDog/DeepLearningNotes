from itertools import count
import gc
import time
import numpy as np
from visdom import Visdom
from agent.main import Agent
from emulator.main import Account

# viz = Visdom()
# assert viz.check_connection()


env = Account()
state = env.reset()
image_shape = state.shape
print(image_shape)

agent = Agent(image_shape, 3)

max_episodes = 1000
global_step = 0
for episode in range(max_episodes):
    state = env.reset()
    cache_reward = []
    cache_valaue = []
    while True:
        global_step += 1
        action = agent.get_epsilon_policy(np.expand_dims(state, 0))
        next_state, reward, done = env.step(action)
        agent.append(state, action, reward, done, next_state)

        cache_reward.append(reward)
        cache_valaue.append(env.total_value)

        if global_step > 1024 and global_step % 32 == 0:
            agent.optimize()

        if done:
            gc.collect()
            break
    print(episode, env.total_value)
