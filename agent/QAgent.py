#BD 2024

# Application of Q-learning to playing of Battleship game
# Done as project for CS351 Intro to AI at UHart

# This particular class is for use setting up the Player agent(s) for 5x5 (sub-)boards.
# This is intended to be used in conjunction with a larger 10x10 board and three other
#   identical agents trained on the other quadrants.

# TODO: Go through and do proper exception handling


import numpy as np
import os
import random

import Config
import StateConversion as sc


# Define a class for use in training a Q-Learning AI agent on a section of a Battleship board
class QAgent:

    def __init__(self, enemy_board = np.array((Config.num_cells), dtype = "int8"),
        discount_factor = Config.discount_factor, learn_rate = Config.learn_rate, epochs = Config.epochs, name = "qt",
        epsilon_max = Config.epsilon_max, epsilon_min = Config.epsilon_min, decay_rate = Config.decay_rate):

        ## Hyperparameters
        # discount factor (0<discount_factor<=1), importance of future rewards
        self.discount_factor = discount_factor  # slightly less than 1, so that quicker wins are preferred over longer ones
        # learning rate
        self.learn_rate = learn_rate
        # used to handle importance of minimizing number of turns, i.e. so quicker wins are preferred
        #self.turn_count_subtract_rate = 0.02
        # set the decay constant used for determining explore vs exploit
        self.exploration_prob = exploration_prob
        # number of epochs to run through
        self.epochs = epochs
        # name of q-table (for use in saving and loading)
        self.name = name
        # epsilon-greedy policy params
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        # the current value of epsilon
        self.epsilon = epsilon_max


        ## Q-table initialization
        self.q_table = [[] for _ in range(Config.num_q_parts)]#np.zeros((Config.num_q_parts, (Config.num_cell_states ** Config.num_cells / Config.num_q_parts), Config.num_cells), dtype = "int8")
        # Load each partition of the Q-table into their respective portions of the Q-table
        if Config.load_q_table:
            self.load_q_table()
        # otherwise, initialize the Q-table as per the configuration in "Config.py"
        self.q_table = self.new_q_table()


        ## Game state
        # store the current game state and state number
        self.cur_state = []
        self.cur_state_num = 3**9
        # action can only be True or False, so using booleans for that, no var needed to represent states
        # define the board used for representing shots (empty, hit, miss)
        self.shot_board = np.zeros(np.shape(enemy_board), dtype = "int8")
        # represent the basis of truth, i.e. the enemy board (true, false -- depends on whether ship occupies cell or not)
        self.enemy_board = enemy_board
        # used to keep track of the agent's score
        #self.score = 0
        # used to keep track of number of hits thus far
        self.num_hits = 0
        # represent the passage of time by keeping track of turns/shots taken
        self.num_shots = 0
        # number cells occupied by enemy ships
        self.init_num_ships = 0
        for cell in enemy_board:
            if cell:
                self.init_num_ships += 1
        # since the Q-max depends on the number of cells and the weight of hits, it's set to hit_weight * num_cells
        #   this is the hypothetical maximum, however it is unacheivable in many scenarios, i.e. when not every single cell
        #   is going to be a hit
        self.q_max_possible = Config.hit_weight * self.init_num_ships * (self.discount_factor ** self.init_num_ships)

        ## Agent game interaction
        # represent the set of all possible board states given the enemy board
        self.possible_boards = []
        self.gen_possible_boards()  # and generate them
        # translate current action to board coordinate (each element corresponds to an index of self.shot_board)
        self.cur_actions = np.zeros(np.shape(self.shot_board), dtype = Config.cell_state_dtype)
        # current action to be taken
        self.cur_action = np.int8(0)
        # next state number
        self.next_state_num = 0
        # represent the next state
        self.next_state = self.cur_state
        # represent the current Q-max for the next state
        self.q_max = np.max(self.q_table[self.next_state_num])


    # calculate the new q value for a given state/action pair (as inferred by given coords for an action)
    def calc_new_q_val(self):
        return (1 - self.learn_rate) * (self.q_table[self.state_num][self.cur_action]) + self.learn_rate * self.calc_q_val_bellman_at_cell()


    # calculate q-value using Bellman's equation
    #   i.e. q(s, a) = R(s, a) + gamma * q_max(s', a')
    # returns q-value for given state-action pair, as gathered by a given coordinate
    #   (i.e. the discounted estimate optimal q-value of next state)
    def calc_q_val_bellman_at_cell(self):
        return self.calc_reward_at_cell(self.cur_action) + self.discount_factor * self.q_max


    # calculate the immediate reward for taking an action (shot) at a given state
    # (represents "R(s, a)" in Bellman's equation
    def calc_reward_at_cell(self):
        # if it's a hit, reward the agent
        if self.enemy_board[self.cur_actions[self.cur_action]]:
            self.num_hits += 1
            return Config.hit_weight
        # if it's a miss, punish the agent
        return Config.miss_weight

    
    # Train the Q-table for the given number of epochs on the given data
    def train(self):
        print("\nBeginning training process for Q-table %s." % self.name)
        # go through the process for the given number of epochs
        for cur_epoch in range(self.epochs):
            print("\tTable:\t%s\tEpoch:\t%d" % (self.name, cur_epoch))
            self.do_epoch()
        print("\nDone training Q-table %s!" % self.name)
        
    
    # run through an entire epoch
    def do_epoch(self):
        # set up the new epoch, starting at a randomly set board state
        self.new_epoch()

        # go through until the game is won
        for i in range(np.shape(self.cur_state)[0]):    # max num of turns is length of board
            # check if in win state
            if self.num_hits == self.num_ships:
                # if so, done with epoch, break from loop
                print("\t\tCurrent epoch done.")
                break

            # take next step
            self.step()


    # set things up for a new epoch
    def new_epoch(self):
        # update epsilon
        self.update_epsilon(cur_epoch)
        
        # select and set a new state randomly out of the set of possible states for the current enemy board
        self.set_state(sc.state_to_num(random.choice(self.possible_boards)))

        # count the number of hits and shots already taken (shots taken is equivalent to turn count)
        self.count_board()


    # count the number of hits and shots already taken (shots is equivalent to turn count)
    def count_board(self):
        # reset hits and shots taken
        self.num_hits = 0
        self.num_shots = 0
        # go through the entire board state, adding to num_hits and num_shots appropriately
        for i in self.cur_state:
            # if it's not 0, a shot was taken at this cell
            if i:
                self.num_shots += 1
                # and if it's a 1, it's a hit
                if i == 1:
                    self.num_hits += 1


    # update epsilon, so as to decrease the amount of exploration relative to exploitation over time
    def update_epsilon(self, cur_epoch = 1):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.decay_rate * cur_epoch)

    
    # take the next step in the training process
    def step(self, action_selection = "epsilon-greedy"):
        # determine the action to go with
        if action_selection == "epsilon-greedy":
            self.choose_action_epsilon_greedy()
        else:
            self.choose_action_greedy()

        # update the q-table
        self.q_table[self.cur_state_num][self.cur_action]

        # update q-max
        self.q_max = np.max(self.q_table[self.next_state_num])
        # set the state of the board to the next state
        self.set_state(self.next_state_num)


    # determine the next action to use via an epsilon-greedy policy
    def choose_action_epsilon_greedy(self):
        # pick a random num from 0 to 1, and check if it's larger than epsilon. if so, exploit
        if np.random.rand() > self.epsilon:
            self.cur_action = self.q_max
        # otherwise, explore
        else:
            # pick a random action from the set of available actions
            self.cur_action = random.choice(self.cur_actions)
        self.update_action()

    
    # determine the next action via a greedy policy (for use in evaluation)
    def choose_action_greedy(self):
        self.cur_action = self.q_max
        self.update_action()
    

    # update the agent's state given the choice of current action
    def update_action(self):
        self.calc_next_state()
    

    # set the state of the board to the given state
    def set_state(self, state_num = 3**9):
        # set the current state
        self.cur_state_num = state_num
        self.cur_state = sc.num_to_state(state_num, Config.num_cell_states, Config.num_cells)
        # set the current actions options
        self.set_actions()


    # get the set of possible actions as an array of indices pertaining to the index of the board on which that action applies
    def set_actions(self):
        # hold the set of actions
        self.cur_actions = []
        # go through each of the cells of the current state
        for i in range(np.shape(self.cur_state)[0]):
            # if the cell is empty, then it's a possible action; add its index to the set of possible actions
            if not self.cur_state[i]:
                self.cur_actions.append(i)
    

    # get the next state
    def calc_next_state(self):
        # initialize the next state to the current state
        self.next_state = np.copy(self.cur_state)
        # using the current action chosen, figure out what the next state would be based on the enemy board
        index = self.cur_actions[self.cur_action]
        # if it's a hit, set the next state accordingly
        if self.enemy_board[index]:
            self.next_state[index] = 1
        # otherwise it's a miss, set accordingly
        else:
            self.next_state[index] = 2
        # set the next state number
        self.next_state_num = sc.state_to_num(self.next_state)

    
    # set enemy board helper function (for use training on multiple different board states in the same session)
    def set_enemy_board(self, new_enemy_board = np.zeros((16), dtype = Config.cell_state_dtype)):
        print("\nSetting new enemy board...")
        self.enemy_board = new_enemy_board
        print("Enemy board set.\n")
        # generate all possible board states for the board
        self.gen_possible_boards()
        
        # reset init_num_ships
        self.init_num_ships = 0
        # count the number of cells with a ship
        for i in new_enemy_board:
            if i:
                self.init_num_ships += 1



    # generate the set of possible board states given the current enemy board
    def gen_possible_boards(self):
        print("\nGenerating possible boards for the given enemy board...")
        # reset possible_boards
        self.possible_boards = []
        # go through the range of possible states, which correspond to all possible combinations of shots taken, which for a 4x4 board is 2^16
        for i in range(2**Config.num_cells):
            # represent the number i as a base 2 board state
            cur_shots_state = sc.num_to_state(i, 2, Config.num_cells) * 2   # * 2 to say they're all misses prior to checking
            
            # for any 0s in the current state, replace the corresponding cell in self.enemy_board to 0
            for i in range(np.shape(cur_shots_state)[0]):
                # check if any of the shots fired are hits. if so, reflect it in the board state.
                if self.enemy_board[i]:
                    cur_shots_state[i] = 1
            
            # add the current state to the list of currently possible board states
            self.possible_boards.append(cur_shots_state)
        
        print("Done generating possible enemy boards.\n")
                
 
    # Testing a new idea for lowering number of actions per state dynamically
    #   (with this version, the previously used q-table partitioning functions are no longer functional without modification)
    def new_q_table(self):
        print("\nInitializing new Q-table...")
        # store the new Q-table as a multidimensional list, where each element contains all states pertaining to a given turn
        new_table = [[] for _ in range(Config.num_cells)]

        # go through for each possible state of the board
        for cur_state in range(Config.primes_list[Config.num_cell_states - 2] ** Config.num_cells):
            # determine and temporarily store the current turn, as found by the number of empty (0) cells in the current state
            cur_turn = -1       # starting at -1 to offset
            for i in sc.num_to_state(cur_state):
                if not i:
                    cur_turn += 1

            # state is full, can't do anything, so, no actions
            if cur_turn == -1:
                continue
            
            # add to element in new_table pertaining to the turn of the current state
            new_table[cur_turn].append(np.zeros((Config.num_cells - cur_turn), dtype = Config.q_value_dtype))
        
        #print("Lengths of table:")
        #for turn in range(len(new_table)):
        #    print("\tTurn %d:\t%d" % (turn, len(new_table[turn])))
        print("New Q-table initialized.\n")

        # return the new table as a list of ndarrays of shape(partition_length, num_cells)
        return [np.array(new_table[i], dtype = Config.q_value_dtype) for i in range(Config.num_q_parts)]



    # Save the current version of the q-table to local storage device
    def save_q_table(self, name = "qt"):
        # notify the user of the process in the console
        print("\nSaving the Q-table to local disk...")
        # check to see if files already exist in the save directory
        if os.path.isfile(Config.qt_save_dir, name + "_table.npy"):
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
            # Save it as a numpy file
            np.save(name + "_table.npy", self.q_table, allow_pickle = False)
            print("\nQ-table saved to local disk.\n")
        except Exception as e:
            print("QuadrantAgent.py: save_q_table(): EXCEPTION saving Q-table:\n\t%s" % str(e))
        
        return True
                

    # Load the Q-table from storage
    def load_q_table(self, name = "qt"):
        # check if partitions exist within the save dierctory
        qt_isparts = self.parts_exist(name)
        # handle if q-table partitions not found
        if not os.path.isfile(Config.qt_save_dir, name + "_table.npy"):
            print("Q-Table file(s) not found. Using fallback (initialize new empty Q-Table)...")
            self.q_table = self.new_q_table()
            return False
        
        try:
            print("Q-Table file(s) found. Loading...")
            # Load it from the numpy files
            np.load(name + "_table.npy", self.q_table, allow_pickle = False)
            print("\nQ-table loaded from local disk.\n")
        except Exception as e:
            print("QuadrantAgent.py: load_q_table(): EXCEPTION loading Q-table:\n\t%s" % str(e))
        
        return True








    ## UNUSED: (temporarily kept here for reference)
"""
    # Initialize a new Q-table based on the configuration options set in Config.py
    def old_new_q_table(self):
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
                new_table[cur_part].append(np.zeros((Config.num_cells), dtype = Config.cell_state_dtype))
        
        # return the new table as a list of ndarrays of shape(partition_length, num_cells)
        return [np.array(new_table[i], dtype = Config.cell_state_dtype) for i in range(Config.num_q_parts)]

    
    # convert action to board coordinate (in consideration of how action will be related to the number of empty cells in a board state)
    def get_nth_action_index(self, action = 0, state = sc.num_to_state(3**9)):
        # get the nth 0 in given state, where n is action. if it doesn't exist, return -1, denoting there are no more 0s
        for i in range(len(state)):
            if not state[i]:
                action -= 1
            if action == 0:
                return i
        
        return -1


    # get the maximum possible q-value for the all possible actions after the given state
    def calc_max_q(self):
        # initialize the current max possible score to the current score
        cur_max = self.score
        # figure out max based on if the rest of the required shots are done perfectly
        for i in range(self.init_num_ships - self.num_hits):    # need to take one turn per ship-occupied cell
            # treating i as turn count
            true_max += 1 - (i * self.turn_count_subtract_rate)

        return cur_max


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
    def save_q_table_old(self, name = "qt"):
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
    def load_q_table_old(self, name = "qt"):
        # check if partitions exist within the save dierctory
        qt_isparts = self.parts_exist(name)
        # handle if q-table partitions not found
        if False in qt_isparts:
            print("Q-Table partition(s) not found. Using fallback (initialize new empty Q-Table)...")
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
"""
