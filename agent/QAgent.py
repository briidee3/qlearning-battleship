#BD 2024

# Application of Q-learning to playing of Battleship game
# Done as project for CS351 Intro to AI at UHart

# This particular class is for use setting up the Player agent(s) for 5x5 (sub-)boards.
# This is intended to be used in conjunction with a larger 10x10 board and three other
#   identical agents trained on the other quadrants.

# TODO: Go through and do proper exception handling


import numpy as np
import os

import Config
import StateConversion as sc


# Define a class for use in training a Q-Learning AI agent on a section of a Battleship board
class QAgent:

    def __init__(self, enemy_board = np.array((Config.num_cells), dtype = "bool"),
        discount_factor = 0.98, learn_rate = 0.8, turn_count_subtract_rate = 0.02, decay_const = 0.001, epochs = 1000):

        ## Hyperparameters
        # discount factor (0<discount_factor<=1), importance of future rewards
        self.discount_factor = discount_factor  # slightly less than 1, so that quicker wins are preferred over longer ones
        # learning rate
        self.learn_rate = learn_rate
        # used to handle importance of minimizing number of turns, i.e. so quicker wins are preferred
        self.turn_count_subtract_rate = 0.02
        # set the decay constant used for determining explore vs exploit
        self.decay_const = decay_const
        # number of epochs to run through
        self.epochs = epochs


        ## Q-table initialization
        self.q_table = [[] for _ in range(Config.num_q_parts)]#np.zeros((Config.num_q_parts, (Config.num_cell_states ** Config.num_cells / Config.num_q_parts), Config.num_cells), dtype = "int8")
        # Load each partition of the Q-table into their respective portions of the Q-table
        if Config.load_q_table:
            self.load_q_table()
        # otherwise, initialize the Q-table as per the configuration in "Config.py"
        self.q_table = self.new_q_table()


        ## Game state
        # action can only be True or False, so using booleans for that, no var needed to represent states
        # define the board used for representing shots (empty, hit, miss)
        #self.shot_board = shot_board
        # represent the basis of truth, i.e. the enemy board (true, false -- depends on whether ship occupies cell or not)
        self.enemy_board = enemy_board
        # used to keep track of the agent's score
        self.score = 0
        # used to keep track of number of hits thus far
        self.num_hits = 0
        # represent the passage of time by keeping track of turns 
        self.cur_turn = 1
        # number cells occupied by enemy ships; initialized to 0, then set by counting number of true states in the enemy board
        #self.init_num_ships = 0
        #for i in range(enemy_board.shape[0]):
        #    for j in range(enemy_board.shape[1]):
        #        if enemy_board[i][j]:
        #            self.q_max += 1
        # since the Q-max depends on the number of cells and the weight of hits, it's set to hit_weight * num_cells
        #   this is the hypothetical maximum, however it is unacheivable in many scenarios, i.e. when not every single cell
        #   is going to be a hit
        self.q_max = Config.hit_weight * Config.num_cells   #self.init_num_ships


    # get the maximum possible q-value for the all possible actions after the given state
    def calc_max_q(self):
        # initialize the current max possible score to the current score
        cur_max = self.score
        # figure out max based on if the rest of the required shots are done perfectly
        for i in range(self.init_num_ships - self.num_hits):    # need to take one turn per ship-occupied cell
            # treating i as turn count
            true_max += 1 - (i * self.turn_count_subtract_rate)

        return cur_max

    # calculate q-value using Bellman's equation
    #   i.e. q(s, a) = R(s, a) + gamma * q_max(s', a')
    # returns q-value for given state-action pair, as gathered by a given coordinate
    def calc_q_val_bellman_at_cell(self, action_loc = np.zeros((1), dtype = "int16")):
        return self.calc_reward_at_cell(action_loc) + self.discount_factor * self.calc_max_q


    # calculate the immediate reward for taking an action (shot) at a given state
    # (represents "R(s, a)" in Bellman's equation
    def calc_reward_at_cell(self, action_loc = np.zeros((1), dtype = "int16")):
        # if the shot has already been taken, nothing can be gained from doing it again, so return 0
        if self.shot_board[action_loc] != states["empty"]:
            return 0
        # otherwise, check if there's a boat there, reward is 1
        elif self.enemy_board[action_loc]:
            return 1
        # otherwise, 0
        else:
            return 0


    # calculate the new q value for a given state/action pair (as inferred by given coords for an action)
    def calc_new_q_val(self, action_loc = np.zeros((1), dtype = "int16")):
        return (1 - self.learn_rate) * (self.q_table[self.shot_board][action_loc]) + self.learn_rate * self.calc_q_val_bellman_at_cell(action_loc)



    # Train the Q-table for the given number of epochs on the given data
    def train(self, data = self.enemy_board, epochs = self.num_turns):
        # go through the process for the given number of epochs
        for cur_epoch in range(epochs):
            return


    # Initialize a new Q-table based on the configuration options set in Config.py, as well as those passed to the class initializer
    def new_q_table(self):
        # get the partition cutoffs given the current config
        part_cutoffs = Config.part_cutoffs
        # store the new Q-table as a multidimensional list
        new_table = [[] for _ in range(Config.num_q_parts)]

        # go through for each part of the Q-table
        for cur_part in range(Config.num_q_parts):
            # get the range of states (in number form) to be used in the current partition
            cur_range = range(part_cutoffs[0])
            # if not the first partition, range is end of prev partition to end of current
            if cur_part != 0:
                cur_range = range(part_cutoffs[cur_part - 1], part_cutoffs[cur_part])

            # go through all states in current partition
            for cur_state in cur_range:
                # create a NxN board of int8s to represent the q values for the current board state
                #   where N = width (or height, since it's a square) of the board
                #   (represented as a 1D array of length N^2)
                new_table[cur_part].append(np.array((Config.num_cells), dtype = Config.cell_state_dtype))
        
        # return the new table as a list of ndarrays of shape(partition_length, num_cells)
        return [np.array(new_table[i], dtype = Config.cell_state_dtype) for i in range(Config.num_q_parts)]
                

    # Select a partition of the Q-table based on the given state number
    def get_part(self, num = 0):
        # Return the index of the partition containing the given state number based on the sizes and constraints of the partitions
        for i in range(Config.num_q_parts):
            # check if number is less than the current cutoff value
            if num < Config.part_cutoffs[i]:
                # if so, it's in the current partition; return its index
                return i
        
        # partition not found, return -1
        print("QuadrantAgent.py: WARNING: Partition not found in get_part(%d)." % num)
        return -1


    # Save the current version of the q-table to local storage device
    def save_q_table(self, name = "qt"):
        # notify the user of the process in the console
        print("Saving the Q-table to local disk...")
        # check to see if files already exist in the save directory
        if True in self.parts_exist(name):
            # Notify the user if there already exists the pertaining files, and give them the option to
            #   cancel the saving process to prevent overwriting an existing table
            save_check = '1'
            while save_check:
                if save_check != '1':
                    print("\nPlease input \"y\" (yes) or (default) \"n\" (no).")
                save_check = input("WARNING: Files already exist in the designated partition storage directory.\nContinue anyways? (y/N)")
                if save_check[0] == 'y' or save_check[0] == 'Y':
                    print("Saving Q-table...")
                    break
            # handle if the user would like to cancel the saving process
            if save_check[0] != 'y':
                print("Saving process canceled.")
                return False
        
        try:
            # go through each of the partitions of the current Q-table, save them to separate files
            for i in range(Config.num_q_parts):
                # save the current partition
                #with open(os.path.join(Config.qt_parts_dir, name + ".p%d" % i), "w") as cur_part:
                #    cur_part.write = ":".join(map(str, self.q_table[i]))
                #    cur_part.close()
                # Save it as a numpy file
                np.save(name + "_p%d.npy" % i, self.q_table[i], allow_pickle = False)
            print("\nQ-table saved to local disk.")
        except Exception as e:
            print("QuadrantAgent.py: save_q_table(): EXCEPTION saving Q-table:\n\t%s" % str(e))
        
        return True
                

    # Load the Q-table from storage
    def load_q_table(self, name = "qt"):
        # check if partitions exist within the save dierctory
        qt_isparts = self.parts_exist(name)
        # handle if q-table partitions not found
        if False in qt_isparts:
            print("Q-Table partition %d not found. Using fallback (initialize new empty Q-Table)...")
            self.q_table = self.new_q_table()
            return False
        
        try:
            # otherwise, load the Q-table partitions
            for i in range(Config.num_q_parts):
                # load the current partition into self.q_table
                #with open(os.path.join(Config.qt_parts_dir, name + ".p%d" % i), "r") as cur_part:
                #    self.q_table[i] = json.loads(cur_part.read().split(":"))
                #    cur_part.close()
                # Load it from the numpy files
                np.load(name + "_p%d.npy" % i, self.q_table[i], allow_pickle = False)
        except Exception as e:
            print("QuadrantAgent.py: load_q_table(): EXCEPTION loading Q-table:\n\t%s" % str(e))
        
        return True


    # Check if partitions exist in the Q-table partitions save direcrtory
    def parts_exist(self, name = "qt"):
        # boolean list to denote if Q-table partitions are all available. initially assume it's not available
        qt_isparts = [False for _ in range(Config.num_q_parts)]
        # check to ensure each partition is there prior to loading
        for i in range(Config.num_q_parts):
            # set boolean for current partition denoted in qt_isparts
            if os.path.isfile(os.path.join(Config.qt_parts_dir, name + "_p%d.npy" % i)) :
                qt_isparts[i] = True
        
        return qt_isparts

