import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class Swish(nn.Module):
    """Relu激活函数变种"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mul(x, torch.sigmoid(x))


class DQN(nn.Module):
    def __init__(self, image_shape, output_size):
        super().__init__()
        in_channels, in_height, in_width = image_shape
        # Relu变种激活函数
        self.swish = Swish()
        # 卷积层
        self.filter1 = nn.Conv2d(in_channels, out_channels=32,
                                 kernel_size=(8, 8), stride=(4, 4))
        self.filter2 = nn.Conv2d(in_channels=32, out_channels=64,
                                 kernel_size=(4, 4), stride=(2, 2))
        self.filter3 = nn.Conv2d(in_channels=64, out_channels=64,
                                 kernel_size=(3, 3), stride=(1, 1))
        # MLP
        filter_output_shape = self._get_filter_size(image_shape)
        self.linear = nn.Linear(filter_output_shape, 512)
        self.linear_output = nn.Linear(512, output_size)

    def forward(self, x):
        return self.linear_output(
            self._build_common_network(x))

    def _get_filter_size(self, shape):
        # 获取卷积层输出size
        batch = 1
        x = Variable(torch.rand(batch, *shape))
        net = self.filter3(self.filter2(self.filter1(x)))
        return int(np.prod(net.size()[1:]))

    def _build_common_network(self, x):
        net = self.swish(self.filter1(x))
        net = self.swish(self.filter2(net))
        net = self.swish(self.filter3(net))
        net = net.view(net.size(0), -1)
        return self.swish(self.linear(net))


class DuelingDQN(nn.Module):
    def __init__(self, image_shape, output_size):
        super().__init__()
        in_channels, in_height, in_width = image_shape
        # Relu变种激活函数
        self.swish = Swish()
        # 卷积层
        self.filter1 = nn.Conv2d(in_channels, out_channels=32,
                                 kernel_size=(8, 8), stride=(4, 4))
        self.filter2 = nn.Conv2d(in_channels=32, out_channels=64,
                                 kernel_size=(4, 4), stride=(2, 2))
        self.filter3 = nn.Conv2d(in_channels=64, out_channels=64,
                                 kernel_size=(3, 3), stride=(1, 1))
        # MLP
        filter_output_shape = self._get_filter_size(image_shape)
        self.linear = nn.Linear(filter_output_shape, 512)
        # 输出层
        self.linear_value = nn.Linear(512, output_size)
        self.linear_advantage = nn.Linear(512, 1)

    def forward(self, x):
        net = self._build_common_network(x)
        value = self.linear_value(net)
        advantage = self.linear_advantage(net)
        return value + (advantage - torch.mean(advantage))

    def _get_filter_size(self, shape):
        # 获取卷积层输出size
        batch = 1
        x = Variable(torch.rand(batch, *shape))
        net = self.filter3(self.filter2(self.filter1(x)))
        return int(np.prod(net.size()[1:]))

    def _build_common_network(self, x):
        net = self.swish(self.filter1(x))
        net = self.swish(self.filter2(net))
        net = self.swish(self.filter3(net))
        net = net.view(net.size(0), -1)
        return self.swish(self.linear(net))












