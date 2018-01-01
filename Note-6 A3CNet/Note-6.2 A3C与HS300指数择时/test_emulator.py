import numpy as np
from emulator.main import Account


A = Account()
A.reset()
for i in range(1000):
    action = np.random.randint(0, 3, 1)
    A.step(action)
