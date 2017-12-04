import numpy as np
import tensorflow as tf
from FCNet import FCNet

inputs = tf.placeholder(tf.float32, [None, 5], 'inputs')
targets = tf.placeholder(tf.int32, [None], 'targets')

actor = FCNet('FCNet')
actor_predict = actor(inputs, 3)

loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=actor_predict))

l2_loss = 0.01 * tf.reduce_mean(actor.get_regularizers())

cost = loss + l2_loss
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

train_data = np.random.uniform(size=(32, 5))
train_target = np.random.randint(0, 3, size=[32], dtype=np.int32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _, c, l, l2 = sess.run([optimizer, cost, loss, l2_loss],
                               {inputs:train_data, targets:train_target})
        print (i, c, l, l2)