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


# compile one table from all existing tables, and each of their rotational permutations
def rot_tables():
    sum_table = np.memmap("./agent/q_table/tot_sum.np", dtype = "float32", mode = "r+", shape = (3**16, 16))

    tables = []
    for i in range(4):
        tables.append(np.memmap("./agent/q_table/qt_%s_table.np" % str(i), dtype = "float32", mode = "r+", shape = (3**16, 16)))

    # go through each rotational permutation for each state-action space
    for state in range(3**16):
        for r in range(4):  # can have 4 different rotations using rot90 per state-action space
            cur_rot_state = sc.num_to_state(state).reshape(4,4)
            for i in range(r):
                cur_rot_state = np.rot90(cur_rot_state)
            cur_rot_state = cur_rot_state.reshape(16)
            cur_state_num = sc.state_to_num(cur_rot_state)
            
            for table in tables:
                sum_table[cur_state_num] += table[cur_state_num] 



if __name__ == "__main__":
    optimizing_test()
