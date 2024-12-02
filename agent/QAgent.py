#BD 2024

# Application of Q-learning to playing of Battleship game
# Done as project for CS351 Intro to AI at UHart

# This particular class is for use setting up the Player agent(s) for 5x5 (sub-)boards.
# This is intended to be used in conjunction with a larger 10x10 board and three other
#   identical agents trained on the other quadrants.

# TODO: Go through and do proper exception handling


from pathlib import Path
import multiprocessing as mp
import numpy as np
import os
import sys
import random

from . import Config
from . import StateConversion as sc


# Define a class for use in training a Q-Learning AI agent on a section of a Battleship board
class QAgent:

    def __init__(self, enemy_board = np.array((Config.num_cells), dtype = "int8"),
        discount_factor = Config.discount_factor, learn_rate = Config.learn_rate, epochs = Config.epochs, name = "qt",
        epsilon_max = Config.epsilon_max, epsilon_min = Config.epsilon_min, decay_rate = Config.decay_rate, memmap = False):

        # check if output should be muted, and if so, mute it
        if Config.mute_qa:
            sys.stdout = open(os.devnull, 'w')

        ## Hyperparameters
        # discount factor (0<discount_factor<=1), importance of future rewards
        self.discount_factor = discount_factor  # slightly less than 1, so that quicker wins are preferred over longer ones
        # learning rate
        self.learn_rate = learn_rate
        # used to handle importance of minimizing number of turns, i.e. so quicker wins are preferred
        #self.turn_count_subtract_rate = 0.02b
        # number of epochs to run through
        self.epochs = epochs
        # name of q-table (for use in saving and loading)
        self.name = name
        # epsilon-greedy policy params (explore vs exploit)
        self.epsilon_max = epsilon_max  # max/initial epsilon val
        self.epsilon_min = epsilon_min  # min possible epsilon val (decreases over time)
        self.decay_rate = decay_rate    # decay constant 
        # the current value of epsilon
        self.epsilon = epsilon_max
        # save the initial value of epsilon
        self.epsilon_init = epsilon_max


        ## Q-table initialization
        # file name for storage of q-table
        self.filename = os.path.join(Config.qt_save_dir, name + "_table.np")
        # for use denoting whether or not to use memory mapping for q-table
        self.memmap = memmap
        # if not, create/load normally
        self.q_table = []
        if not self.memmap:
            self.q_table = [[] for _ in range(Config.num_q_parts)]#np.zeros((Config.num_q_parts, (Config.num_cell_states ** Config.num_cells / Config.num_q_parts), Config.num_cells), dtype = "int8")
            # Load each partition of the Q-table into their respective portions of the Q-table
            if Config.load_q_table:
                self.load_q_table()
        # otherwise, check to see if memmap file already exists. if not, make it
        elif not os.path.isfile(self.filename):
            self.q_table = np.memmap(self.filename, dtype = Config.q_value_dtype, mode = "w+", shape = ((Config.num_cell_states ** Config.num_cells), Config.num_cells))
            # immediately write to disk for access from other processes
            self.q_table.flush()
            del self.q_table
            # reopen in read/write mode
            self.q_table = self.memmap_load_qt()
        # otherwise, load the memory mapped file pertaining to the q-table name from storage
        else:
            self.q_table = self.memmap_load_qt()


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
        #for cell in enemy_board:
        #    if cell:
        #        self.init_num_ships += 1
        # since the Q-max depends on the number of cells and the weight of hits, it's set to hit_weight * num_cells
        #   this is the hypothetical maximum, however it is unacheivable in many scenarios, i.e. when not every single cell
        #   is going to be a hit
        self.q_max_possible = Config.hit_weight * self.init_num_ships * (self.discount_factor ** self.init_num_ships)

        ## Agent game interaction
        # represent the set of all possible board states given the enemy board
        self.possible_boards = []
        #self.gen_possible_boards()  # and generate them
        # translate current action to board coordinate (each element corresponds to an index of self.shot_board)
        self.cur_actions = np.zeros(np.shape(self.shot_board), dtype = Config.cell_state_dtype)
        # current action to be taken
        self.cur_action = np.int8(0)
        # next state number
        self.next_state_num = 0
        # represent the next state
        self.next_state = self.cur_state
        # represent the current Q-max for the next state
        self.q_max = np.float32(0)


        ## Evaluation data
        # keep track of the average ratio of hits to misses
        self.hit_count = 0
        self.miss_count = 0
        # keep track of the average new q value
        self.sum_new_q = np.float64(0)
        self.q_count = 0
        # keep track of avg reward
        self.sum_rewards = 0

    
    # Initialize the Q-table
    def init(self):
        # check if using memory map
        if not self.mmap:
            self.q_table = self.new_q_table()
        else:
            self.q_table = self.mmap_load_qt()

    
    # set enemy board helper function (for use training on multiple different board states in the same session)
    def set_enemy_board(self, new_enemy_board = np.zeros((16), dtype = Config.cell_state_dtype)):
        #print("\n" + self.name + ": Setting new enemy board...")
        self.enemy_board = np.copy(new_enemy_board)
        #print(self.name + ": Enemy board set.\n")
        #print(new_enemy_board)
        # generate all possible board states for the board
        if Config.mode == "train":
            self.gen_possible_boards()
        
        # reset init_num_ships
        self.init_num_ships = 0
        # count the number of cells with a ship
        for i in new_enemy_board:
            if i:
                self.init_num_ships += 1
        #print(self.init_num_ships)

    
    # Get the action that the agent should use during evaluation for the given board state (greedy)
    #def get_action(self):
        # get the max q for the current board state
    #    cur_max_q = self.q_table[sc.state_to_num(self.enemy_board)]


    # calculate the new q value for a given state/action pair (as inferred by given coords for an action)
    def calc_new_q_val(self):
        # calculate next q value, clipping the bellman portion to help stave off divergence
        q = self.q_table[self.cur_state_num][self.cur_action] + self.learn_rate * (self.calc_q_val_bellman_at_cell() - self.q_table[self.cur_state_num][self.cur_action])
        # iterate evaluation data
        self.q_count += 1
        self.sum_new_q += np.float64(q)

        return q


    # calculate q-value using Bellman's equation
    #   i.e. q(s, a) = R(s, a) + gamma * q_max(s', a')
    # returns q-value for given state-action pair, as gathered by a given coordinate
    #   (i.e. the discounted estimate optimal q-value of next state)
    def calc_q_val_bellman_at_cell(self):
        return self.calc_reward_at_cell() + self.discount_factor * self.q_max


    # calculate the immediate reward for taking an action (shot) at a given state
    # (represents "R(s, a)" in Bellman's equation
    def calc_reward_at_cell(self):
        # if it's a hit, reward the agent
        if self.enemy_board[self.cur_action] != 0:
            self.hit_count += 1
            self.sum_rewards += Config.hit_weight
            return Config.hit_weight
        # if it's a miss, punish the agent
        self.miss_count += 1
        self.sum_rewards += Config.miss_weight
        return Config.miss_weight

    
    # Train the Q-table for the given number of epochs on the given data
    def train(self):
        print("\n" + self.name + ": Beginning training process for Q-table %s." % self.name)
        # reset epsilon
        self.epsilon = self.epsilon_init
        # go through the process for the given number of epochs
        for cur_epoch in range(self.epochs):
            #print("\tTable:\t%s\tEpoch:\t%d" % (self.name, cur_epoch))
            self.do_epoch(cur_epoch)
        print("\n" + self.name + ": Done training Q-table %s!" % self.name)


    # Evaluate the Q-table after training
    def eval(self):
        avg_hit_miss = (self.hit_count / self.miss_count)
        avg_q = (self.sum_new_q / self.q_count)
        avg_reward = (self.sum_rewards / (self.hit_count + self.miss_count))
        print("\n" + self.name + ": Evaluation:\n")
        print("\t" + self.name + ": Average ratio of hits/misses:\t" + str(avg_hit_miss))
        print("\t" + self.name + ": Average new Q-value:\t" + str(avg_q))
        print("\t" + self.name + ": Average reward:\t" + str(avg_reward) + "\n")
        if Config.save_stats:
            with open(os.path.join(Config.qt_save_dir, self.name + "_LR" + str(self.learn_rate) + "_DF" + str(self.discount_factor) + "_DR" + str(self.decay_rate) + ".txt"), "w") as save:
                save.write(":".join(map(str, [avg_hit_miss, avg_q, avg_reward])))
                save.close()
        
    
    # run through an entire epoch
    def do_epoch(self, cur_epoch):
        # set up the new epoch, starting at a randomly set board state
        self.new_epoch(cur_epoch)

        # go through until the game is won
        for i in range(np.shape(self.cur_state)[0]):    # max num of turns is length of board
            # check if in win state
            if self.num_hits == self.init_num_ships:
                # if so, done with epoch, break from loop
                #print("\t\t" + self.name + ": Current epoch done.")
                break

            # take next step
            self.step()


    # set things up for a new epoch
    def new_epoch(self, cur_epoch):
        # update epsilon
        self.update_epsilon(cur_epoch)
        
        # generate new boards until all ship cells haven't already been hit
        # select and set a new state randomly out of the set of possible states for the current enemy board
        init_state = sc.state_to_num(random.choice(self.possible_boards))
        self.set_state(init_state)
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

        # update q-max of next step
        self.q_max = np.max(self.q_table[self.next_state_num])
        # update the q-table
        self.q_table[self.cur_state_num][self.cur_action] = self.calc_new_q_val()

        # set the state of the board to the next state
        self.set_state(state_num = self.next_state_num)


    # determine the next action to use via an epsilon-greedy policy
    def choose_action_epsilon_greedy(self):
        # pick a random num from 0 to 1, and check if it's larger than epsilon. if so, exploit
        if np.random.rand() > self.epsilon:
            # set cur_action to the first location of q_max (which at this point in runtime should be q_max of current state)
            self.cur_action = np.argmax(self.q_table[state_num])
        # otherwise, explore
        else:
            # pick a random action from the set of available actions
            self.cur_action = random.choice(self.cur_actions)
        self.calc_next_state()

    
    # determine the next action via a greedy policy (for use in evaluation)
    def choose_action_greedy(self):
        self.cur_action = np.argmax(self.q_table[state_num])
        self.calc_next_state()

    
    # get the max of the q table at the current state
    #def get_q_max(self, state_num = self.cur_state_num):
    #    return np.argmax(self.q_table[state_num])
    

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
            if self.cur_state[i] == 0:
                self.cur_actions.append(i)


    # get the next state during a game
    def calc_next_state(self):
        # initialize the next state to the current state
        self.next_state = np.copy(self.cur_state)

        # if it's a hit, set the next state accordingly
        if self.enemy_board[self.cur_action] == 1:
            self.next_state[self.cur_action] = 1
            self.num_hits += 1
        # otherwise it's a miss, set accordingly
        else:
            self.next_state[self.cur_action] = 2
        # set the next state number
        self.next_state_num = sc.state_to_num(self.next_state)




    # generate the set of possible board states given the current enemy board
    def gen_possible_boards(self):
        print("\n" + self.name + ": Generating possible boards for the given enemy board...")
        # reset possible_boards
        self.possible_boards = []
        # go through the range of possible states, which correspond to all possible combinations of shots taken, which for a 4x4 board is 2^16
        for i in range(2**Config.num_cells):
            # represent the number i as a base 2 board state
            cur_shots_state = sc.num_to_state(i, 2, Config.num_cells) * 2   # * 2 to say they're all misses prior to checking
            # for any 0s in the current state, replace the corresponding cell in self.enemy_board to 0
            for i in range(np.shape(self.enemy_board)[0]):
                # check if any of the shots fired are hits. if so, reflect it in the board state.
                if self.enemy_board[i] != 0 and cur_shots_state[i] == 2:
                    cur_shots_state[i] = 1
            
            # add the current state to the list of currently possible board states
            self.possible_boards.append(cur_shots_state)
        
        print(self.name + ": Done generating possible enemy boards.\n")

    
    # manually set the q-table
    def set_q_table(self, q_table):
        self.q_table = q_table

    
    # get the q table
    def get_q_table(self):
        return self.q_table


    # Initialize a new Q-table based on the configuration options set in Config.py
    def new_q_table(self):
        # get the partition cutoffs given the current config
        #part_cutoffs = Config.part_cutoffs
        # store the new Q-table as a multidimensional list
        return np.zeros(((Config.num_cell_states ** Config.num_cells), Config.num_cells), dtype = Config.q_value_dtype)


    # Save the current version of the q-table to local storage device
    def save_q_table(self):
        # notify the user of the process in the console
        print("\n" + self.name + ": Saving the Q-table to local disk...")
        # check to see if files already exist in the save directory
        if os.path.isfile(os.path.join(Config.qt_save_dir, self.name + "_table.npy")):
            # Notify the user if there already exists the pertaining files, and give them the option to
            #   cancel the saving process to prevent overwriting an existing table
            save_check = '1'
            while save_check:
                if save_check != '1':
                    print("\n" + self.name + ": Please input \"y\" (yes) or (default) \"n\" (no).")
                save_check = input("WARNING: Files already exist in the designated partition storage directory. Continue anyways? (y/N)")
                if save_check[0] == 'y' or save_check[0] == 'Y':
                    print("Saving Q-table...")
                    break
            # handle if the user would like to cancel the saving process
            if save_check[0] != 'y':
                print(self.name + ": Saving process canceled.")
                return False
        
        try:
            # Save it as a numpy file
            np.save(os.path.join(Config.qt_save_dir, self.name + "_table.npy"), self.q_table, allow_pickle = False)
            print("\n" + self.name + ": Q-table saved to local disk.\n")
        except Exception as e:
            print(self.name + ": QuadrantAgent.py: save_q_table(): EXCEPTION saving Q-table:\n\t%s" % str(e))
        
        return True
                

    # Load the Q-table from storage
    def load_q_table(self):
        # handle if q-table partitions not found
        if not os.path.isfile(os.path.join(Config.qt_save_dir, self.name + "_table.npy")):
            print(self.name + ": Q-Table file(s) not found. Using fallback (initialize new empty Q-Table)...")
            self.q_table = self.new_q_table()
            return False
        
        try:
            print(self.name + ": Q-Table file(s) found. Loading...")
            # Load it from the numpy files
            self.q_table = np.load(os.path.join(Config.qt_save_dir, self.name + "_table.npy"), allow_pickle = False)
            print("\n" + self.name + ": Q-table loaded from local disk.\n")
        except Exception as e:
            print(self.name + ": QuadrantAgent.py: load_q_table(): EXCEPTION loading Q-table:\n\t%s" % str(e))
            return False
        return True


    # save q tableusing memory map
    def memmap_save_qt(self, lock = mp.Lock()):
        with lock:
            self.q_table.flush()
    

    # load q table using memory map
    def memmap_load_qt(self):
        return np.memmap(self.filename, dtype = Config.q_value_dtype, mode = "r+", shape = ((Config.num_cell_states ** Config.num_cells), Config.num_cells))







    ## UNUSED: (temporarily kept here for reference)
"""
                
 
    # Testing a new idea for lowering number of actions per state dynamically
    #   (with this version, the previously used q-table partitioning functions are no longer functional without modification)
    def old_new_new_q_table(self):
        print("\n" + self.name + ": Initializing new Q-table...")
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
        print(self.name + ": New Q-table initialized.\n")

        # return the new table as a list of ndarrays of shape(partition_length, num_cells)
        return [np.array(new_table[i], dtype = Config.q_value_dtype) for i in range(len(new_table))]




    

    
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
