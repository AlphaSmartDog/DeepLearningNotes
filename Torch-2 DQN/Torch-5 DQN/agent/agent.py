from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax
from torch.autograd import Variable
from agent.forward import DQN
from agent.access import Access


# Ensure values are greater than epsilon to avoid numerical instability.
_EPSILON = 1e-6


class Agent(object):
    def __init__(self, image_shape, output_size,
                 capacity=int(1e6), learning_rate=1e-3):
        self.output_size = output_size
        self.access = Access(capacity)
        self.value_net = DQN(image_shape, output_size)
        self.target_net = deepcopy(self.value_net)
        # 自动使用gpu
        self.gpu = torch.cuda.is_available()
        if self.gpu:
            self.value_net.cuda()
            self.target_net.cuda()

        self.optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def get_deterministic_policy(self, x):
        x = Variable(torch.from_numpy(x.astype(np.float32)))
        if not self.gpu:
            out = self.value_net(x).data.numpy()
            return np.argmax(out, axis=1)
        else:
            x = x.cuda()
            out = self.value_net(x)
            out = out.cpu().data.numpy()
            return np.argmax(out, axis=1)

    def get_stochastic_policy(self, x):
        x = Variable(torch.from_numpy(x.astype(np.float32)))
        if not self.gpu:
            out = softmax(self.value_net(x), 1)
            out = out.data.numpy()
            return np.random.choice(self.output_size, 1, p=out[0])[0]
        else:
            x = x.cuda()
            out = softmax(self.value_net(x), 1)
            out = out.cpu().data.numpy()
            return np.random.choice(self.output_size, 1, p=out[0])[0]

    def get_epsilon_policy(self, x, epsilon=0.9):
        if np.random.uniform() > epsilon:
            return np.random.randint(self.output_size)
        else:
            return self.get_stochastic_policy(x)

    def optimize(self, batch_size=64, gamma=.9):
        batch = self.sample(batch_size)
        if self.gpu:
            state, action, reward,  done, next_state = \
                [Variable(torch.from_numpy(np.float32(i))).cuda() for i in batch]
            action = action.type(torch.LongTensor).cuda()
        else:
            state, action, reward,  done, next_state = \
                [Variable(torch.from_numpy(np.float32(i))) for i in batch]
            action = action.type(torch.LongTensor)

        value = self.value_net(state).gather(1, action.unsqueeze(1))
        next_value = self.target_net(next_state).detach()
        next_value = next_value.max(1)[0].view([-1, 1])
        value = value.squeeze(1)
        next_value = next_value.squeeze(1)
        target = done * reward + (1 - done) * (reward + gamma * next_value)
        loss = self.loss_func(value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_target(self):
        # update target network parameters
        for t, s in zip(self.target_net.parameters(), self.value_net.parameters()):
            t.data.copy_(s.data)

    def append(self, *args):
        self.access.append(*args)

    def sample(self, batch_size=128):
        return self.access.sample(batch_size)



