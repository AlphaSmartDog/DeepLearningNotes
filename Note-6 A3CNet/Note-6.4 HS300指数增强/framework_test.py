import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
from env.main import Account
from agent.main import Access, Framework


env = Account()
init = env.reset()
print(init.shape)


name = 'W0'
input_shape = [381, 240, 58]
action_size = 3
A = Access(input_shape, action_size)
W0 = Framework(name, A, input_shape, action_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    W0.run(sess, max_episodes=2, t_max=3)