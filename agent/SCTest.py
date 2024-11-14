# BD 2024

# Short test of StateConversion 

import numpy as np
import StateConversion as sc


init_state = np.zeros((25), dtype="int8")

for i in range(0, 25):
    init_state[i] = int(np.random.rand() * 3 - 0.00001)

print(init_state)
init_num = sc.stateToNum(init_state)
print(init_num)
final_state = sc.numToState(init_num)
print(final_state)

