import numpy as np
import torch
from torch.autograd import Variable
from agent.forward import DQN


np_state = np.random.normal(size=(50, 16, 50, 58)).astype(np.float32)
torch_state = Variable(torch.from_numpy(np_state), volatile=False)

image_shape = (16, 50, 58)
forward = DQN(image_shape, 3)
print(forward(torch_state))