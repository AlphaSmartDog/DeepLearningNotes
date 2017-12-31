import tensorflow as tf
from agent.framework import Framework
from agent.main import Agent
from emulator.main import Account


# env = Account()
# state = env.reset()
#
# agent = Framework()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
#
# while True:
#     action = agent.get_stochastic_policy(sess, state)
#     next_state, reward, done = env.step(action)
#     agent.update_cache(state, action, reward, next_state, done)
#     state = next_state
#     if done:
#         break
#
# agent.update_value_net(sess)
# agent.update_target_net(sess)


env = Account()
state = env.reset()

agent = Agent()
while True:
    action = agent.get_stochastic_policy(state)
    next_state, reward, done = env.step(action)
    agent.update_cache(state, action, reward, next_state, done)
    state = next_state
    if done:
        break

agent.update_target()
agent.update_eval()
agent.save_model()
agent.restore_model()






