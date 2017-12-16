import rqdatac
from rqdatac import *

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

import tensorflow as tf

from agent.access import Access
from env.env_main import Account
from agent.actor_critic import Agent

state_size = 58
batch_size = 50
action_size = 3

rqdatac.init('xinjin', '123456', ('172.19.182.162', 16003))

# get data
stock_list = ['000300.XSHG', '000016.XSHG', '000905.XSHG']
start_date = '2012-01-01'
start_date_features = '2011-12-01'
end_date = '2017-06-30'
fields_daily = ['open', 'close']
fields_hf = ['open', 'high', 'low', 'close', 'total_turnover']

daily = get_price(stock_list, start_date, end_date, fields=fields_daily, adjust_type='post', frequency='1d')
high_freq = get_price(stock_list, start_date_features, end_date, fields=fields_hf, adjust_type='post', frequency='15m')

# account
test_env = Account(daily_prices=daily, high_freq_data=high_freq)
print(len(test_env.features))
print(test_env.features[-1].shape)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    with tf.device("/cpu:0"):
        A = Access(batch_size, state_size, action_size)
        W = Agent('W0', A, batch_size, state_size, action_size)
        A.restore(sess, 'model/saver_4.ckpt')
        W.init_or_update_local(sess)
        env = test_env
        state = env.reset()
        for _ in range(200):
            action = W.get_deterministic_policy_action(sess, state)
            state, reward, done = env.step(action)

value, reward = env.plot_data()

print(pd.Series(value))

print(pd.Series(reward))
print(pd.Series(np.zeros_like(reward)).plot(figsize=(16, 6), color='r'))

print('some difference')

pd.Series(value).plot(figsize=(16, 6))

pd.Series(reward).plot(figsize=(16, 6))
pd.Series(np.zeros_like(reward)).plot(figsize=(16, 6), color='r')
