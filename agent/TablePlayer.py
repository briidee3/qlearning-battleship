# BD 2024
# Load and control a Q-learning agent to play battleship with user or monte carlo using trained Q-tables


import numpy as np
import os, pathlib

from . import QAgent as qa
from . import StateConversion as sc
from . import Config


class TablePlayer:

    def __init__(self):
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

        # data trackers
        self.turn = 0
        self.num_shots = 0
        self.num_hits = 0


    # reset stats of the agent
    def reset(self):
        self.turn = 0
        self.num_shots = 0
        self.num_hits = 0

    
    # initialize the q tables
    def init_agents(self):
        for i in range(self.q_steps):
            self.q_tables.append(qa.QAgent())


    # Load the stored Q-tables
    def load_tables(self):
        # reset list
        self.q_tables = []
        # go through for each of the quadrants/files in "q_tables"
        for table in os.listdir(Config.qt_save_dir):
            self.q_tables.append(np.memmap(os.path.join(Config.qt_save_dir, table), dtype = Config.q_value_dtype, mode = "r+", shape = ((Config.num_cell_states ** Config.num_cells), Config.num_cells)))
    

    # Set the macro board state for the agent to use
    def set_enemy_board_state(self, board):
        self.enemy_macro_board = board

        # reset 
        self.enemy_slices = []
        self.agent_slices = []
        # set the slices for each agent
        for i in range(int(self.q_steps / 2)):
            for j in range(int(self.q_steps / 2)):
                self.agent_slices.append(np.zeros((16), dtype = Config.cell_state_dtype))
                self.enemy_slices.append(np.ndarray.flatten(self.enemy_macro_board[i*self.step_size : self.step_size+i*self.step_size, j*self.step_size : self.step_size+j*self.step_size]))


    
    # Get the max qs for each of the board quadrants
    def get_q_max(self):
        # reset the lists
        self.max_q_actions = []
        self.max_q_vals = []
        # go thru each agent
        for i in range(self.q_steps):
            # get num for current state
            cur_state_num = sc.state_to_num(self.agent_slices[i])
            # add index of max q
            self.max_q_actions.append(np.argmax(self.q_tables[i][cur_state_num]))
            # add max q
            self.max_q_vals.append(self.q_tables[i][cur_state_num][self.max_q_actions[i]])

        # return the index pertaining to the q-table with max q val
        q_max = np.argmax(self.max_q_vals)
        # if max is 0, pick one of them at random
        if q_max == 0:
            q_max = np.random.randint(16)

        return q_max

    
    # get the next board state with regards to the given q-table, action, current state indices
    def get_next_board_state(self, agent = 0, action = 0):
        # adjust to taste
        action -= 1
        # check if it's a hit
        if self.enemy_slices[agent][action] == 1:
            # set the corresponding location on the agent's board view to a hit
            self.agent_slices[agent][action] = 1
            self.num_hits += 1
        # otherwise it's a miss
        else:
            # set the location to a miss
            self.agent_slices[agent][action] = 2
            self.num_shots += 1


    # take a step
    def step(self, coords = True):
        # get the index of the agent with the max q for the current board state
        cur_agent = self.get_q_max()
        # get the next micro-board state for that agent, update that quadrant of the board
        self.get_next_board_state(cur_agent, self.max_q_actions[cur_agent])

        # return the current action for use in evaluation
        action = self.max_q_actions[cur_agent]
        coords = [int(action / 4) + cur_agent * 4, (action % 4) + cur_agent * 4]

        if coords:
            return coords
        else:
            return coords[0] * 8 + coords[1]
        
        
