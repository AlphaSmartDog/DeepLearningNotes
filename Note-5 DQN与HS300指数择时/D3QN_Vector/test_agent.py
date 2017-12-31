import numpy as np
import tensorflow as tf
from agent.framework import Framework
from emulator_v0.main import Account

A = Account()
F = Framework()
# print(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
state, universe = A.reset()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

order = F.get_deterministic_policy(sess, state)
next_state, next_universe, reward, done, value, portfolio = \
    A.step(order, universe)

for i in range(2048):
    F.update_cache(state, order, reward, next_state, done)

F.update_value_net(sess)

