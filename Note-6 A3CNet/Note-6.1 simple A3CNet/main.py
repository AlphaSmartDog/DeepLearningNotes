import multiprocessing
import threading
import tensorflow as tf
from Access import Access
from Framework import ExplorerFramework


NUMS_CPU = multiprocessing.cpu_count()
state_size = 4
action_size = 2


tf.reset_default_graph()
sess = tf.Session()
with tf.device("/cpu:0"):
    A = Access(state_size, action_size)
    F_list = []
    for i in range(NUMS_CPU):
        F_list.append(ExplorerFramework(A, 'W%i' % i, state_size, action_size))

    COORD = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    threads_list = []
    for ac in F_list:
        job = lambda: ac.run(sess)
        t = threading.Thread(target=job)
        t.start()
        threads_list.append(t)
    COORD.join(threads_list)