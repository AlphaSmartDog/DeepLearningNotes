from agent.noise import Noise

n = Noise(4)
for _ in range(128):
    print(n())