import multiprocessing
import threading
import tensorflow as tf
from agent.main import Access, Framework


# NUMS_CPU = multiprocessing.cpu_count()
NUMS_CPU = 1
inputs_shape = [381, 240, 58]
action_size = 3
max_episodes = 1


GD = {}


class Worker(Framework):
    def __init__(self, name, access, inputs_shape, action_size):
        super().__init__(name, access, inputs_shape, action_size)

    def run(self, sess, max_episodes, t_max=8):
        episode_score_list = []
        episode = 0
        while episode < max_episodes:
            episode += 1
            episode_socre, _ = self.run_episode(sess, t_max)
            episode_score_list.append(episode_socre)
            GD[str(self.name)] = episode_score_list
            if self.name == 'W0':
                print('Episode: %f, score: %f' % (episode, episode_socre))
                print('\n')


with tf.Session() as sess:
    with tf.device("/cpu:0"):
        A = Access(inputs_shape, action_size)
        F_list = []
        for i in range(NUMS_CPU):
            F_list.append(Worker('W%i' % i, A, inputs_shape, action_size))
        COORD = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()

        threads_list = []
        for ac in F_list:
            job = lambda: ac.run(sess, max_episodes)
            t = threading.Thread(target=job)
            t.start()
            threads_list.append(t)
        COORD.join(threads_list)
        A.save(sess, 'model/saver_1.ckpt')