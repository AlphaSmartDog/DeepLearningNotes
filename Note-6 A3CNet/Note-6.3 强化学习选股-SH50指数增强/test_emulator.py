import numpy as np
from emulator.main import Account


A = Account()
A.reset()
for i in range(200):
    order = np.random.randint(0, 3, 50)
    A.step(order)
    print(A.quote.total_value)
