# BD 2024
# Configuration options and default hyperparameters for the Q-Learning agent to be used
#   in playing the Battleship board game.


## HYPERPARAMETERS

# number of epochs to be trained
epochs = 1000
# learn rate of the agent
learn_rate = 0.8
# discount factor, denoting importance of future rewards
discount_factor = 0.98

# exploration parameters for use with epsilon-greedy policy
epsilon_max = 1.0
epsilon_min = 0.05
decay_rate = 0.001



## OPTIONS

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
q_value_dtype = "float16"

# define shot board states as integers representing one of three states
#   defined outside of class since is invariant
shot_states = {
            "empty":    0,
            "hit":      1,
            "miss":     2
        }

# set the weights for hits and misses
hit_weight = 6        # hit
miss_weight = -1        # miss

# used to denote whether or not to load an existing Q-table from local storage
load_q_table = False
# local directory in which the Q-table partitions are stored
qt_save_dir = "q_table"
# number of partitions to separate the q-table into
num_q_parts = 1
# Represent the maximums for each q-table partition
part_cutoffs = []
for i in range(num_q_parts):
    part_cutoffs.append(int(primes_list[num_cell_states - 2] ** (num_cells - num_cells * ((num_q_parts - (i + 1)) / num_q_parts))))
