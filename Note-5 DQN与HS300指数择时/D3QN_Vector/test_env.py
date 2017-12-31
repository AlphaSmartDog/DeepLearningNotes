# import numpy as np
# from emulator.market import Market
# from emulator.trading import Trading
#
# M = Market()
# T = Trading()
# T.reset()
# for i in range(468):
#     state, universe = M.step(i)
#     order = np.random.randint(0, 3, size=50)
#     reward, done, total_value = T.step(order, universe, i)
#     print(reward, done)

# from emulatorVer0.main import Account
# import numpy as np
#
# A = Account()
# state, universe = A.reset()
# for i in range(467):
#     order = np.random.randint(0, 3, size=50)
#     state, universe, reward, done, value, portfolio\
#         = A.step(order, universe)
#     print(i, reward, value)


from emulator_v0.trading import dataset_open
print(dataset_open.iloc[22:])