import gc
import gym
from agent.agent import Agent


MAX_EPISODES = 5


env = gym.make('BipedalWalker-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

agent = Agent(state_size, action_size)
state = env.reset()
for _ in range(int(1024)):
    action = agent(state) + agent.get_noise()
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
        action = agent(state) + agent.get_noise()
        next_state, reward, done, info = env.step(action)
        agent.append(state, action, reward, done, next_state)

        state = next_state
        agent.optimize()

        if done:
            break