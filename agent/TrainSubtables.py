# BD 2024
# Train the quadrant agents on each quadrant of a 8x8 battleship board.


from . import Config
from . import QAgent as qa

import numpy as np
import multiprocessing as mp
import os
from pathlib import Path


# for use accessing q_table save directory, which is local to the agent directory
path = Path(__file__).parent


# Define a class to use for management of training Q-tables in parallel
class TrainSubtables:

    def __init__(self, board_shape = (8, 8), step_size = 4, num_iterations = 1):
        # initialize board state
        self.board_state = np.zeros((8, 8), dtype = Config.cell_state_dtype)
        # set the number of times to go through the process of generating a new board and training the agents
        self.num_iterations = num_iterations
        # define step size for row and col
        self.step_size = step_size
        # get the number of q-steps (i.e. q-table quadrants) to create
        self.num_qsteps = int((board_shape[0] / self.step_size) + (board_shape[1] / self.step_size))
        # create the q-tables for each qstep
        self.agents = []
        for i in range(self.num_qsteps):
            self.agents.append(qa.QAgent(name = "qt_%d" % i))
        # the processes to be used for training
        self.processes = []

    
    # Generate a random ship placement for use in training
    def set_board_state(self, board = np.zeros((8, 8), dtype = Config.cell_state_dtype)):
        self.board_state = board


    # Train the given Q-learning agent
    def train_agent(self, agent, table_slice):
        # initialize the q-table of the agent
        agent.init()
        # set the board for the agent
        agent.set_enemy_board(np.ndarray.flatten(table_slice))
        # train the agent
        agent.train()
        # evaluate agent
        agent.eval()
        # once done, save the agent locally
        #agent.save_q_table()


    # Run through the process of spinning up processes and training agents
    def run_training_processes(self):

        slices = []
        # set the slices for each agent
        for i in range(int(self.num_qsteps / 2)):
            for j in range(int(self.num_qsteps / 2)):
                slices.append(self.board_state[i*self.step_size : self.step_size+i*self.step_size, j*self.step_size : self.step_size+j*self.step_size])

        # reset processes list
        self.processes = []
        # initializing the training processes
        for i in range(1):#len(self.agents)):
            # append new process to processes list and start them
            self.processes.append(mp.Process(target = self.train_agent, args = (self.agents[i], slices[i])))
            self.processes[i].start()

        # wait for the processes to finish
        for proc in self.processes:
            proc.join()


    # to be used for identification of optimal hyperparameters for QAgents
    def optimize(self):
        # processes list
        processes = []
        # agents list
        agents = []
        # board slice to use for testing
        board_slice = self.board_state[0:4, 0:4]

        # define range for each var
        range_learn_rate = [0.05, 0.95]
        range_discount_factor = [0.05, 0.95]
        range_decay_rate = [0.00001, 0.001]

        # go through and run trials for each var
        #learn rate and discount factor
        for i in np.arange(0.05, 1, 0.05):
            agents.append(qa.QAgent(name = "qt_lr_" + str(i), learn_rate = i))
            agents.append(qa.QAgent(name = "qt_df_" + str(i), discount_factor = i))
        
        #decay rate
        for i in np.arange(0.00001, 0.0011, 0.00001):
            agents.append(qa.QAgent(name = "qt_dr_" + str(i), decay_rate = i))
        
        # add them all to individual processes
        for i in range(len(agents)):
            processes.append(mp.Process(target = self.train_agent, args = (agents[i], board_slice)))
            processes[i].start()

        # wait for processes to finish
        for proc in processes:
            proc.join()
