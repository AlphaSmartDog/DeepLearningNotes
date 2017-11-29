import tensorflow as tf
from agent.access import Access
from agent.actor_critic import Agent
from env.env_main import Account

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

state_size = 58
batch_size = 50
action_size = 3


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    with tf.device("/cpu:0"):
        A = Access(batch_size, state_size, action_size)
        W = Agent('W0', A, batch_size, state_size, action_size)
        A.restore(sess,'model/saver_1.ckpt')
        W.init_or_update_local(sess)
        env = Account()
        state = env.reset()
        for _ in range(200):
            action = W.get_deterministic_policy_action(sess, state)
            state, reward, done = env.step(action)

value, reward = env.plot_data()

print(pd.Series(value))

print(pd.Series(reward))
print(pd.Series(np.zeros_like(reward)).plot(figsize=(16,6), color='r'))

# pd.Series(value).plot(figsize=(16,6))
#
# pd.Series(reward).plot(figsize=(16,6))
# pd.Series(np.zeros_like(reward)).plot(figsize=(16,6), color='r')
