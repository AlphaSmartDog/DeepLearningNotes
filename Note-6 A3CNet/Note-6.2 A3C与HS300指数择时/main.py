import multiprocessing
import threading
import tensorflow as tf
from Access import Access
from Framework import ExplorerFramework


NUMS_CPU = multiprocessing.cpu_count()
state_size = [50, 58, 5]
action_size = 3
max_episodes = 100
GD = {}


class Worker(ExplorerFramework):
    def __init__(self, access, name, observation, action_size):
        super().__init__(access, name, observation, action_size)

    def run(self, sess, max_episodes, t_max=32):
        episode_score_list = []
        episode = 0
        while episode < max_episodes:
            episode += 1
            episode_socre = self.run_episode(sess, t_max)
            episode_score_list.append(episode_socre)
            GD[str(self.name)] = episode_score_list
            if self.name == 'W0':
                print('Episode: %f, score: %f' % (episode, episode_socre))
                print('\n')


with tf.Session() as sess:
    with tf.device("/cpu:0"):
        A = Access(state_size, action_size)
        F_list = []
        for i in range(NUMS_CPU):
            F_list.append(Worker(A, 'W%i' % i, state_size, action_size))
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
