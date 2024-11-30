# BD 2024
# Class for use running the training processes 

import numpy as np
import agent.Config as cfg
import agent.TrainSubtables as ts


# test board for initial testing of the traning processes
test_board = np.array([[0, 0, 1, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0, 1, 1],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0]], dtype = cfg.cell_state_dtype)

# test the training process
def test():
    print("Initializing trainer...")
    trainer = ts.TrainSubtables(step_size = 4, num_iterations = 1)
    print("Trainer initialized.")
    


if __name__ == "__main__":
    test()
