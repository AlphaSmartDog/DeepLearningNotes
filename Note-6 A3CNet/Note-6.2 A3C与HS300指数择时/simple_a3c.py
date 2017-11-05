import tensorflow as tf
import numpy as np
import threading
import gym
import os
from scipy.misc import imresize


def copy_src_to_dst(from_scope, to_scope):
    """Creates a copy variable weights operation

    Args:
        from_scope (str): The name of scope to copy from
            It should be "global"
        to_scope (str): The name of scope to copy to
            It should be "thread-{}"

    Returns:
        list: Each element is a copy operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def pipeline(image, new_HW=(80, 80), height_range=(35, 193), bg=(144, 72, 17)):
    """Returns a preprocessed image

    (1) Crop image (top and bottom)
    (2) Remove background & grayscale
    (3) Reszie to smaller image

    Args:
        image (3-D array): (H, W, C)
        new_HW (tuple): New image size (height, width)
        height_range (tuple): Height range (H_begin, H_end) else cropped
        bg (tuple): Background RGB Color (R, G, B)

    Returns:
        image (3-D array): (H, W, 1)
    """
    image = crop_image(image, height_range)
    image = resize_image(image, new_HW)
    image = kill_background_grayscale(image, bg)
    image = np.expand_dims(image, axis=2)

    return image


def resize_image(image, new_HW):
    """Returns a resized image

    Args:
        image (3-D array): Numpy array (H, W, C)
        new_HW (tuple): Target size (height, width)

    Returns:
        image (3-D array): Resized image (height, width, C)
    """
    return imresize(image, new_HW, interp="nearest")


def crop_image(image, height_range=(35, 195)):
    """Crops top and bottom

    Args:
        image (3-D array): Numpy image (H, W, C)
        height_range (tuple): Height range between (min_height, max_height)
            will be kept

    Returns:
        image (3-D array): Numpy image (max_H - min_H, W, C)
    """
    h_beg, h_end = height_range
    return image[h_beg:h_end, ...]


def kill_background_grayscale(image, bg):
    """Make the background 0

    Args:
        image (3-D array): Numpy array (H, W, C)
        bg (tuple): RGB code of background (R, G, B)

    Returns:
        image (2-D array): Binarized image of shape (H, W)
            The background is 0 and everything else is 1
    """
    H, W, _ = image.shape

    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    cond = (R == bg[0]) & (G == bg[1]) & (B == bg[2])

    image = np.zeros((H, W))
    image[~cond] = 1

    return image


def discount_reward(rewards, gamma=0.99):
    """Returns discounted rewards

    Args:
        rewards (1-D array): Reward array
        gamma (float): Discounted rate

    Returns:
        discounted_rewards: same shape as `rewards`

    Notes:
        In Pong, when the reward can be {-1, 0, 1}.

        However, when the reward is either -1 or 1,
        it means the game has been reset.

        Therefore, it's necessaray to reset `running_add` to 0
        whenever the reward is nonzero
    """
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add

    return discounted_r


class A3CNetwork(object):

    def __init__(self, name, input_shape, output_dim, logdir=None):
        """Network structure is defined here

        Args:
            name (str): The name of scope
            input_shape (list): The shape of input image [H, W, C]
            output_dim (int): Number of actions
            logdir (str, optional): directory to save summaries
                TODO: create a summary op
        """
        with tf.variable_scope(name):
            self.states = tf.placeholder(tf.float32, shape=[None, *input_shape], name="states")
            self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
            self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
            self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")

            action_onehot = tf.one_hot(self.actions, output_dim, name="action_onehot")
            net = self.states

            with tf.variable_scope("layer1"):
                net = tf.layers.conv2d(net,
                                       filters=16,
                                       kernel_size=(8, 8),
                                       strides=(4, 4),
                                       name="conv")
                net = tf.nn.relu(net, name="relu")

            with tf.variable_scope("layer2"):
                net = tf.layers.conv2d(net,
                                       filters=32,
                                       kernel_size=(4, 4),
                                       strides=(2, 2),
                                       name="conv")
                net = tf.nn.relu(net, name="relu")

            with tf.variable_scope("fc1"):
                net = tf.contrib.layers.flatten(net)
                net = tf.layers.dense(net, 256, name='dense')
                net = tf.nn.relu(net, name='relu')

            # actor network
            actions = tf.layers.dense(net, output_dim, name="final_fc")
            self.action_prob = tf.nn.softmax(actions, name="action_prob")
            single_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)

            entropy = - self.action_prob * tf.log(self.action_prob + 1e-7)
            entropy = tf.reduce_sum(entropy, axis=1)

            log_action_prob = tf.log(single_action_prob + 1e-7)
            maximize_objective = log_action_prob * self.advantage + entropy * 0.005
            self.actor_loss = - tf.reduce_mean(maximize_objective)

            # value network
            self.values = tf.squeeze(tf.layers.dense(net, 1, name="values"))
            self.value_loss = tf.losses.mean_squared_error(labels=self.rewards,
                                                           predictions=self.values)

            self.total_loss = self.actor_loss + self.value_loss * .5
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=.99)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.gradients = self.optimizer.compute_gradients(self.total_loss, var_list)
        self.gradients_placeholders = []

        for grad, var in self.gradients:
            self.gradients_placeholders.append((tf.placeholder(var.dtype, shape=var.get_shape()), var))
        self.apply_gradients = self.optimizer.apply_gradients(self.gradients_placeholders)

        if logdir:
            loss_summary = tf.summary.scalar("total_loss", self.total_loss)
            value_summary = tf.summary.histogram("values", self.values)

            self.summary_op = tf.summary.merge([loss_summary, value_summary])
            self.summary_writer = tf.summary.FileWriter(logdir)


class Agent(threading.Thread):

    def __init__(self, session, env, coord, name, global_network, input_shape, output_dim, logdir=None):
        """Agent worker thread

        Args:
            session (tf.Session): Tensorflow session needs to be shared
            env (gym.env): Gym environment
            coord (tf.train.Coordinator): Tensorflow Queue Coordinator
            name (str): Name of this worker
            global_network (A3CNetwork): Global network that needs to be updated
            input_shape (list): Required for local A3CNetwork (H, W, C)
            output_dim (int): Number of actions
            logdir (str, optional): If logdir is given, will write summary
                TODO: Add summary
        """
        super(Agent, self).__init__()
        self.local = A3CNetwork(name, input_shape, output_dim, logdir)
        self.global_to_local = copy_src_to_dst("global", name)
        self.global_network = global_network

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = env
        self.sess = session
        self.coord = coord
        self.name = name
        self.logdir = logdir

    def print(self, reward):
        message = "Agent(name={}, reward={})".format(self.name, reward)
        print(message)

    def play_episode(self):
        self.sess.run(self.global_to_local)

        states = []
        actions = []
        rewards = []

        s = self.env.reset()
        s = pipeline(s)
        state_diff = s

        done = False
        total_reward = 0
        time_step = 0
        while not done:

            a = self.choose_action(state_diff)
            s2, r, done, _ = self.env.step(a)

            s2 = pipeline(s2)
            total_reward += r

            states.append(state_diff)
            actions.append(a)
            rewards.append(r)

            state_diff = s2 - s
            s = s2

            if r == -1 or r == 1 or done:
                time_step += 1

                if time_step >= 5 or done:
                    self.train(states, actions, rewards)
                    self.sess.run(self.global_to_local)
                    states, actions, rewards = [], [], []
                    time_step = 0

        self.print(total_reward)

    def run(self):
        while not self.coord.should_stop():
            self.play_episode()

    def choose_action(self, states):
        """
        Args:
            states (2-D array): (N, H, W, 1)
        """
        states = np.reshape(states, [-1, *self.input_shape])
        feed = {
            self.local.states: states
        }

        action = self.sess.run(self.local.action_prob, feed)
        action = np.squeeze(action)

        return np.random.choice(np.arange(self.output_dim) + 1, p=action)

    def train(self, states, actions, rewards):
        states = np.array(states)
        actions = np.array(actions) - 1
        rewards = np.array(rewards)

        feed = {
            self.local.states: states
        }

        values = self.sess.run(self.local.values, feed)

        rewards = discount_reward(rewards, gamma=0.99)
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)

        advantage = rewards - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        feed = {
            self.local.states: states,
            self.local.actions: actions,
            self.local.rewards: rewards,
            self.local.advantage: advantage
        }

        gradients = self.sess.run(self.local.gradients, feed)

        feed = []
        for (grad, _), (placeholder, _) in zip(gradients, self.global_network.gradients_placeholders):
            feed.append((placeholder, grad))

        feed = dict(feed)
        self.sess.run(self.global_network.apply_gradients, feed)


def main():
    try:
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        coord = tf.train.Coordinator()

        save_path = "checkpoint/model.ckpt"
        n_threads = 4
        input_shape = [80, 80, 1]
        output_dim = 3  # {1, 2, 3}
        global_network = A3CNetwork(name="global",
                                    input_shape=input_shape,
                                    output_dim=output_dim)

        thread_list = []
        env_list = []

        for id in range(n_threads):
            env = gym.make("Pong-v0")

            if id == 0:
                env = gym.wrappers.Monitor(env, "monitors", force=True)

            single_agent = Agent(env=env,
                                 session=sess,
                                 coord=coord,
                                 name="thread_{}".format(id),
                                 global_network=global_network,
                                 input_shape=input_shape,
                                 output_dim=output_dim)
            thread_list.append(single_agent)
            env_list.append(env)

        if tf.train.get_checkpoint_state(os.path.dirname(save_path)):
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(sess, save_path)
            print("Model restored to global")
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
            print("No model is found")

        for t in thread_list:
            t.start()

        print("Ctrl + C to close")
        coord.wait_for_stop()

    except KeyboardInterrupt:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
        saver = tf.train.Saver(var_list=var_list)
        saver.save(sess, save_path)
        print()
        print("=" * 10)
        print('Checkpoint Saved to {}'.format(save_path))
        print("=" * 10)

        print("Closing threads")
        coord.request_stop()
        coord.join(thread_list)

        print("Closing environments")
        for env in env_list:
            env.close()

        sess.close()


if __name__ == '__main__':
    main()
