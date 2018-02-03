import numpy as np
from emulator.main import Account
from agent.agent import Agent


env = Account()
state = env.reset()
print(state.shape)
agent = Agent([5, 50, 58], 3)


for i in range(1440):
    state = np.transpose(state, [2, 0, 1])
    state = np.expand_dims(state, 0)
    action = agent.get_epsilon_policy(state)
    reward, next_state, done = env.step(action)
    agent.append(state, action, reward, done, next_state)
    state = next_state

    print(done, reward)
    if done:
        state = env.reset()
        break

print(agent.sample())
# agent.optimize()