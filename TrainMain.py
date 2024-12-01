# BD 2024
# Class for use running the training processes 

import numpy as np
import multiprocessing as mp

import agent.Config as cfg
import agent.TrainSubtables as ts
import agent.QAgent as qa
from game import create_random_opponent


# ships to be used
ships = [2, 2, 3, 3, 4]
# create new random board
rand_board = create_random_opponent(8, ships).get_board()
print(rand_board)

# test board for initial testing of the traning processes
test_board = np.array([[0, 0, 1, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0]], dtype = cfg.cell_state_dtype)

# test the training process
def test():
    print("Initializing trainer...")
    trainer = ts.TrainSubtables(step_size = 4, num_iterations = 1)
    trainer.set_board_state(test_board)
    print("Trainer initialized.")

    print("Beginning training processes...")
    trainer.run_training_processes()



if __name__ == "__main__":
    test()
