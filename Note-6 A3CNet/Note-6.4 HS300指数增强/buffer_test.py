import numpy as np
import tensorflow as tf
from env.main import Account
from agent.main import Agent, Access, Framework


env = Account()
init = env.reset()
print(init.shape)

buffer_state = [init, init, init, init]


def batch_stack(inputs):
    # gather index
    gather_list = 15 + np.arange(len(inputs))
    # stack
    a = [inputs[0][:-1]]
    b = [i[-1] for i in inputs]
    b = [np.expand_dims(i, axis=0) for i in b]
    return np.vstack(a + b), gather_list


s, g = batch_stack(buffer_state)
print(s.shape)
print(g)
