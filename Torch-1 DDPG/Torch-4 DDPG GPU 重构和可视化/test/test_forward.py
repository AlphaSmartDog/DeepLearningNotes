import numpy as np
import torch
from torch.autograd import Variable
from agent.forward import ActorNet


np_state = np.random.normal(size=24).astype(np.float32)
var_state = Variable(torch.from_numpy(np_state))
print(type(var_state))

A = ActorNet(24, 4)
out = A(var_state)
print(type(out))
print(out.data.shape)

