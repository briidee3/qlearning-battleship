# BD 2024
# Load and control a Q-learning agent to play battleship with user or monte carlo using trained Q-tables


import numpy as np
import os, pathlib

from . import QAgent as qa
from . import StateConversion as sc
from . import Config



class TablePlayer:

    def __init__(self, seed = 31415):
        # set the seed for pseudo-randomized RNG seeding
        self.seed = seed
        np.random.seed(seed)
        # step size for use slicing the board
        self.step_size = 4
        # number of micro-states being used
        self.q_steps = 4
        # represent the overall board state
        self.enemy_macro_board = np.zeros((8,8), dtype = Config.cell_state_dtype)
        # used to reference the q-tables
        self.q_tables = []
        # initialize the q-tables
        self.load_tables()

        # used to hold the micro-board states (i.e. the 4x4 states) for use by the agents
        self.enemy_slices = []      # enemy boards
        self.agent_slices = []      # agent's view
        # used to hold the current max-q values and corresponding actions for the current board state
        self.max_q_actions = []
        self.max_q_vals = []

        # keep track of available actions for each slice
        self.cur_actions = [np.arange(0, 16, dtype = "int8").tolist(), 
                            np.arange(0, 16, dtype = "int8").tolist(), 
                            np.arange(0, 16, dtype = "int8").tolist(), 
                            np.arange(0, 16, dtype = "int8").tolist()]
        print(self.cur_actions[0])
        # data trackers
        self.turn = 0
        self.num_shots = 0
        self.num_hits = 0

        # used for setting the RNG seed between runs, to avoid determinism of one-seed RNG
        #self.seed = 12345


    # reset stats of the agent
    def reset(self):
        self.turn = 0
        self.num_shots = 0
        self.num_hits = 0


    # initialize the q tables
    def init_agents(self):
        for i in range(self.q_steps):
            self.q_tables.append(qa.QAgent())
        #temp = self.q_tables[1]
        #self.q_tables[1] = self.q_tables[2]
        #self.q_tables[2] = temp


    # Load the stored Q-tables
    def load_tables(self):
        # reset list
        self.q_tables = []
        # go through for each of the quadrants/files in "q_tables"
        for table in os.listdir(Config.qt_save_dir):
            self.q_tables.append(np.memmap(os.path.join(Config.qt_save_dir, table), dtype = Config.q_value_dtype, mode = "r+", shape = ((Config.num_cell_states ** Config.num_cells), Config.num_cells)))


    # Set the macro board state for the agent to use
    def set_enemy_board_state(self, board = np.zeros((8,8), dtype = Config.cell_state_dtype)):
        self.enemy_macro_board = board
        #print(board)
        # reset
        self.enemy_slices = []
        self.agent_slices = []
        # set the slices for each agent
        for i in range(int(self.q_steps / 2)):
            for j in range(int(self.q_steps / 2)):
                self.agent_slices.append(np.zeros((16), dtype = Config.cell_state_dtype))
                self.enemy_slices.append(np.ndarray.flatten(self.enemy_macro_board[i*self.step_size : self.step_size+i*self.step_size, j*self.step_size : self.step_size+j*self.step_size]))
        #print(self.enemy_slices)
        #exit()


    # Set the RNG seed
    def set_seed(self, seed = 54321):
        self.seed = seed


    # Get the max qs for each of the board quadrants
    def get_q_max(self):
        # set the seed for changing deterministic output of RNG
        np.random.seed(self.seed)

        # reset the lists
        self.max_q_actions = []
        self.max_q_vals = []
        print(self.cur_actions)
        # go thru each agent
        for i in range(self.q_steps):
            # check if no actions left
            if self.cur_actions[i] != []:
                # get num for current state
                cur_state_num = sc.state_to_num(self.agent_slices[i])
                
                # get current q max index
                cur_q_max_index = np.argmax(self.q_tables[i][cur_state_num])
                # get all others of same q value in action space for current state, if any exist
                cur_same_acts = np.where(self.q_tables[i][cur_state_num] == self.q_tables[i][cur_state_num][cur_q_max_index])
                print(self.cur_actions)
                print(cur_same_acts)
                print(cur_q_max_index)
                # if there's multiple, select one at random from the list of those of the same max,
                #   whilst also checking for their inclusion in the current expected action space.
                if np.shape(cur_same_acts)[1] != 1:
                    cur_q_max_index = np.random.choice(np.intersect1d(np.array(self.cur_actions[i]), cur_same_acts[0]))
                # otherwise append the action found with argmax, ensuring it's within the list of possible actions
                else:
                    cur_q_max_index = np.intersect1d(np.array(self.cur_actions[i]), cur_same_acts[0])[0]
                    # check if cur_max is empty. if so, throw an exception
                    if cur_q_max_index == None:
                        raise ValueError('No action selected.')
                
                # add current max q to max_q_actions
                self.max_q_actions.append(cur_q_max_index)
                

                # add max q
                self.max_q_vals.append(self.q_tables[i][cur_state_num][cur_q_max_index])
            # if empty, put in some fake values to prevent this one from being picked without messing up the ordering
            else:
                self.max_q_actions.append(-3)
                self.max_q_vals.append(-9999999)

        # convert max_q_vals to np array
        print(self.max_q_vals)
        self.max_q_vals = np.array(self.max_q_vals)

        # get the index pertaining to the q-table with max q val
        q_max_index = np.argmax(self.max_q_vals)
        # check if others have same val
        cur_same_qmax = np.where(self.max_q_vals == self.max_q_vals[q_max_index])
        # if multiple, select one at random
        if np.shape(cur_same_qmax)[1] != 1:
            q_max_index = np.random.choice(cur_same_qmax[0])

        # update cur_actions for the selected q-table
        self.cur_actions[q_max_index].remove(self.max_q_actions[q_max_index])

        return q_max_index


    # get the next board state with regards to the given q-table, action, current state indices
    def get_next_board_state(self, agent = 0, action = 0):
        # adjust to taste
        #action -= 1
        # check if it's a hit
        if self.enemy_slices[agent][action] == 1:
            # set the corresponding location on the agent's board view to a hit
            #print(self.agent_slices[agent])
            self.agent_slices[agent][action] = 1
            #print(self.agent_slices[agent])
            self.num_hits += 1
        # otherwise it's a miss
        else:
            # set the location to a miss
           # print("Action: ", self.agent_slices[agent][action])
            self.agent_slices[agent][action] = 2
            #print("Agent slice:", self.agent_slices[agent])
            self.num_shots += 1


    # take a step
    def step(self, coords = True):
        # get the index of the agent with the max q for the current board state
        cur_agent = self.get_q_max()
        # get the next micro-board state for that agent, update that quadrant of the board
        self.get_next_board_state(cur_agent, self.max_q_actions[cur_agent])

        # return the current action for use in evaluation
        action = self.max_q_actions[cur_agent]
        #sprint("Agent, action:", cur_agent, action)
        #print("Board:\n%s\n%s\n%s\n%s" % (str(self.agent_slices[0].reshape(4,4)), str(self.agent_slices[1].reshape(4,4)), str(self.agent_slices[2].reshape(4,4)), str(self.agent_slices[3].reshape(4,4))))
        coords = []
        #print(self.enemy_macro_board)
        #print(self.agent_slices[cur_agent])
        #print(self.enemy_slices[cur_agent])
        #print(cur_agent)
        match cur_agent:
            case 2:
                coords = [int(action / 4) + 4, (action % 4)]
            case 1:
                coords = [int(action / 4), (action % 4) + 4]
            case 3:
                coords = [int(action / 4) + 4, (action % 4) + 4]
            case 0:
                coords = [int(action / 4), (action % 4)]
            case _:
                coords = [int(action / 4), (action % 4)]

        return coords
