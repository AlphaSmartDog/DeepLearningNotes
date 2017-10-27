import numpy as np
import tensorflow as tf
from models.ACNet import ACNet

class Test(ACNet):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)

    def test_step(self, states, actions, R):
        self.train_critic(states, R)
        self.train_actor(states, actions, R)
        func = [self.loss_policy, self.loss_value, self.entropy]
        tmp = self.sess.run(func, {self.state: states, self.a_t: actions, self.R: R})
        print(tmp)

state_size = 5
action_size = 3
Agent = Test(state_size, action_size)

train_data = np.random.uniform(size=(32, 5))
train_action = np.random.randint(0, 3, size=[32], dtype=np.int32)
train_discounted_reward = np.random.rand(32, 1) * 100

print(Agent.predict_policy(train_data))
print(Agent.predict_value(train_data))



for i in range(1000):
    print(i)
    Agent.test_step(train_data, train_action, train_discounted_reward)
    #Agent.train_actor(train_data, train_action, train_discounted_reward)
    #Agent.train_critic(train_data, train_discounted_reward)

