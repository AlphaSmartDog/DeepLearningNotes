import gym
from agent.access import Access


A = Access(1024)

env = gym.make('BipedalWalker-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

state = env.reset()
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)

for _ in range(2048):
    A.append(state, action, reward, done, next_state)

buffer = A.sample(64)
print([type(i) for i in buffer])


