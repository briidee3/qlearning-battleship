# BD 2024
# utility functions for working with tables

import numpy as np
import os, sys, pathlib
import agent.Config as Config


# rotate all action spaces in given q table by 90 degrees counter-clockwise
def rotate(q_table, num_rots):
    for i in range(0, np.shape(q_table)[0]):
        cur_action_space = q_table[i]
        cur_action_space = cur_action_space.reshape((4,4))
        q_table[i] = np.ndarray.flatten(np.rot90(cur_action_space, num_rots))

# rotate all of the q_tables
def rot_tables():
    q_tables = []
    for i in range(4):
        q_tables.append(np.memmap(os.path.join(pathlib.Path(__file__).parent, "agent/q_table/qt_%d_table.np" % i), dtype = Config.q_value_dtype, mode = "r+", shape = ((Config.num_cell_states ** Config.num_cells), Config.num_cells)))
    rotate(q_tables[1], 3)
    rotate(q_tables[2], 1)
    rotate(q_tables[3], 2)
    for i in range(4):
        q_tables[i].flush()



# unrotate all of the q_tables
def unrot_tables():
    q_tables = []
    for i in range(4):
        q_tables.append(np.memmap(os.path.join(pathlib.Path(__file__).parent, "agent/q_table/qt_%d_table.np" % i), dtype = Config.q_value_dtype, mode = "r+", shape = ((Config.num_cell_states ** Config.num_cells), Config.num_cells)))
    rotate(q_tables[1], 1)
    rotate(q_tables[2], 3)
    rotate(q_tables[3], 2)
    for i in range(4):
        q_tables[i].flush()


# sum up the tables
def sum_tables():

    summed_table = np.memmap(os.path.join(pathlib.Path(__file__).parent, "agent/q_table/qt_summed_table.np"), dtype = Config.q_value_dtype, mode = "w+", shape = ((Config.num_cell_states ** Config.num_cells), Config.num_cells))
    for i in range(4):
        print(i)
        summed_table += np.memmap(os.path.join(pathlib.Path(__file__).parent, "agent/q_table/qt_%d_table.np" % i), dtype = Config.q_value_dtype, mode = "r+", shape = ((Config.num_cell_states ** Config.num_cells), Config.num_cells))




if __name__ == "__main__":
    unrot_tables()
    #sum_tables()
