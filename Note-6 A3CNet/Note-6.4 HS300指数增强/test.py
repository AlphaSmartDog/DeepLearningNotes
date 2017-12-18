import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from agent.main import Access, Agent
from env.main import Account

tf.reset_default_graph()
sns.set_style('whitegrid')
%matplotlib


inputs_shape = [381, 240, 58]
action_size = 3


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    with tf.device("/cpu:0"):
        A = Access(inputs_shape, action_size)
        W = Agent('W0', A, inputs_shape, action_size)
        A.restore(sess,'model/saver_1.ckpt')
        W.init_or_update_local(sess)
        env = Account()
        state = env.reset()
        for _ in range(466):
            action = W.get_deterministic_policy_action(sess, state)
            state, reward, done = env.step(action)