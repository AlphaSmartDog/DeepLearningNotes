import numpy as np
from ACNet import ACNet

class Test(ACNet):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)

    def test_step(self, states, actions, R):
        self.train_critic(states, R)
        self.train_actor(states, actions, R)
        func = [self.loss_policy, self.loss_value, self.loss_entropy]
        tmp = self.sess.run(func, {self.inputs: states, self.actions: actions, self.targets: R})
        print(tmp)

state_size = 5
action_size = 3
Agent = Test(state_size, action_size)

train_data = np.random.uniform(size=(32, 5))
train_action = np.random.randint(0, 3, size=[32], dtype=np.int32)
train_discounted_reward = np.random.rand(32) * 100

print(Agent.predict_policy(train_data))
print(Agent.predict_value(train_data))



for i in range(1000):
    print(i, Agent.predict_action(np.expand_dims(train_data[0], axis=0)))
    #Agent.test_step(train_data, train_action, train_discounted_reward)
    #Agent.train_actor(train_data, train_action, train_discounted_reward)
    #Agent.train_critic(train_data, train_discounted_reward)

