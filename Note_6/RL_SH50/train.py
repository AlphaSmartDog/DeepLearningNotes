import rqdatac
from rqdatac import *

import time
import multiprocessing
import threading

import tensorflow as tf

from agent.access import Access
from agent.framework import Framework
from env.env_main import Account

rqdatac.init('xinjin', '123456', ('172.19.182.162', 16003))

# get data
# stock_list = ['000300.XSHG', '000016.XSHG', '000905.XSHG']
stock_list = ['C99', 'CS99', 'A99']
start_date = '2012-01-01'
start_date_features = '2011-12-01'
end_date = '2017-06-30'
fields_daily = ['open', 'close']
fields_hf = ['open', 'high', 'low', 'close', 'total_turnover']

daily = get_price(stock_list, start_date, end_date, fields=fields_daily, adjust_type='post', frequency='1d')
high_freq = get_price(stock_list, start_date_features, end_date, fields=fields_hf, adjust_type='post', frequency='15m')
trading_info = dict()
trading_info['margin_rate'] = [i.margin_rate for i in instruments(stock_list)]
trading_info['contract_multiplier'] = [i.contract_multiplier for i in instruments(stock_list)]

# account
train_env = Account(daily_prices=daily, high_freq_data=high_freq, trading_info=trading_info)
print(len(train_env.features))
print(train_env.features[-1].shape)

tf.reset_default_graph()
# NUMS_CPU = multiprocessing.cpu_count()
NUMS_CPU = 1
state_size = train_env.features[-1].shape[-1]
batch_size = train_env.features[-1].shape[1]
action_size = 3
max_episodes = 50
model_path = 'model/saver_1.ckpt'
restore_model = False

GD = {}
class Worker(Framework):

    def __init__(self, name, access, batch_size, state_size, action_size):
        super().__init__(name, access, batch_size, state_size, action_size)
        self.env = train_env

    def run(self, sess, max_episodes, t_max=8):
        episode_score_list = []
        episode = 0
        while episode < max_episodes:
            episode += 1
            episode_score, _ = self.run_episode(sess, t_max)
            episode_score_list.append(episode_score)
            GD[str(self.name)] = episode_score_list
            if self.name == 'W0':
                print('Episode: %f, score: %f' % (episode, episode_score))
                print('\n')

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# with tf.Session(config=config) as sess:
t_start = time.ctime()
with tf.Session() as sess:
    with tf.device("/cpu:0"):

        A = Access(batch_size, state_size, action_size)
        if restore_model:
            A.restore(sess, model_path)
        F_list = []

        # writer = tf.summary.FileWriter("./model/output", sess.graph)
        # writer.flush()

        for i in range(NUMS_CPU):
            F_list.append(Worker('W%i' % i, A, batch_size, state_size, action_size))

        COORD = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()

        threads_list = []
        for ac in F_list:
            t = threading.Thread(target=lambda: ac.run(sess, max_episodes))
            t.start()
            threads_list.append(t)

        COORD.join(threads_list)

        A.save(sess, model_path)

        # writer.close()

t_end = time.ctime()
