# BD 2024

# Some brief tests

import numpy as np
import os
from pathlib import Path
from . import StateConversion as sc


def sctest():
    init_state = np.zeros((25), dtype="int8")

    for i in range(0, 25):
        init_state[i] = int(np.random.rand() * 3 - 0.00001)

    print(init_state)
    init_num = sc.stateToNum(init_state)
    print(init_num)
    final_state = sc.numToState(init_num)
    print(final_state)


def optimizing_test():
    file_dir = os.path.join(Path(__file__).parent, "q_table")
    print(file_dir)
    data = []

    for file in os.listdir(file_dir):
        if file.__contains__("_dr_"):
            with open(os.path.join(file_dir, file), "r") as f:
                cur_data = f.read().split(":")
                for i in range(len(cur_data)):
                    cur_data[i] = float(cur_data[i])
                data.append(cur_data)
                f.close()
    
    print("Max:", np.max(np.array(data)))

if __name__ == "__main__":
    optimizing_test()
