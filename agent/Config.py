# BD 2024
# Configuration options and default hyperparameters for the Q-Learning agent to be used
#   in playing the Battleship board game.


from pathlib import Path
import os


## HYPERPARAMETERS

# number of epochs to be trained
epochs = 10000
# learn rate of the agent
learn_rate = 0.05
# discount factor, denoting importance of future rewards
discount_factor = 0.1

# exploration parameters for use with epsilon-greedy policy
#epsilon_max = 1.0
epsilon_max = 0.5
epsilon_min = 0.05
decay_rate = 0.0008



## OPTIONS

# mode of operation for the QAgents (train or eval/play)
mode = "train"
# set epsilon and learn_rate to 0 to only exploit current knowledgess during eval
if mode == "eval":
    epsilon = 0
    learn_rate = 0

# number of instances of TrainSubtables processes to be run
num_ts_subprocs = 2
# number of boards each TrainSubtables instance should run through
num_ts_iter = 1
# chunk size for multiprocessing pools
chunk_size = 1
num_runs = 1

# save statistics (used for optimizing hyperparams)
save_stats = False
# whether or not to mute QAgent
mute_qa = False

# for the use case in mind, the 4x4 board has 16 cells
num_cells = 4**2          # aka "n"
# number of possible states per cell
num_cell_states = 3     # aka "m"

# step size for use in hierarchical q-learning
step_size = 2

# list of prime numbers. only using the first two, since m - 1 = 3 - 1 = 2 for our use case, but more can be added for other use cases
primes_list = [2, 3]
# data type of the states being used
cell_state_dtype = "int8"
q_value_dtype = "float32"

# define shot board states as integers representing one of three states
#   defined outside of class since is invariant
shot_states = {
            "empty":    0,
            "hit":      1,
            "miss":     2
        }

# set the weights for hits and misses
hit_weight = 0.05        # hit
miss_weight = -0.01        # miss

# used to denote whether or not to load an existing Q-table from local storage
load_q_table = False
# local directory in which the Q-table partitions are stored
qt_save_dir = os.path.join(Path(__file__).parent, "q_table")
# number of partitions to separate the q-table into
num_q_parts = 1
# Represent the maximums for each q-table partition
part_cutoffs = []
for i in range(num_q_parts):
    part_cutoffs.append(int(primes_list[num_cell_states - 2] ** (num_cells - num_cells * ((num_q_parts - (i + 1)) / num_q_parts))))
