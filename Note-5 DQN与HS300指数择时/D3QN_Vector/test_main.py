from emulator_v0.main import Account
from agent.main import Agent
from config import *


env = Account()
agent = Agent()

# fill cache
for episode in range(5):
    state, universe = env.reset()
    while True:
        action = agent.get_stochastic_policy(state, 0)
        next_state, universe, reward, done, value, portfolio \
            = env.step(action, universe)
        agent.update_cache(state, action, reward, next_state, done)
        state = next_state
        if done:
            break


NUM_EPISODES = 10

episodes_train = []
global_step = 0
for episode in range(NUM_EPISODES):
    state, universe = env.reset()
    episode_step = 0
    while True:
        global_step += 1
        episode_step += 1

        action = agent.get_stochastic_policy(state)
        next_state, universe, reward, done, value, portfolio \
            = env.step(action, universe)
        agent.update_cache(state, action, reward, next_state, done)
        state = next_state

        if global_step % TARGET_STEP_SIZE == 0:
            agent.update_target()

        if episode_step % TRAIN_STEP_SIZE == 0 or done:
            agent.update_eval()

            if done:
                break


