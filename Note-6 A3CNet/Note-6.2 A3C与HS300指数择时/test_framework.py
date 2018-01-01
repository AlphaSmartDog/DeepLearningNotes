from agent.framework import Framework
from agent.access import Access


state_size = [50, 58, 5]
A = Access(state_size, 3)
F = Framework(A, state_size, 3, "W0")