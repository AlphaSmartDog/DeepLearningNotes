import numpy as np
from emulator.main import Account


A = Account()
state = A.reset()
for i in range(1440):
    action = np.random.randint(0, 3)
    reward, next_state, done = A.step(action)
    print(reward)
