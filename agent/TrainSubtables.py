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

    def __init__(self, board_shape = (8, 8), step_size = 4, num_iterations = 1, board_queue = mp.Queue()):

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
            self.agents.append(qa.QAgent(name = "qt_%d" % i, memmap = True))
        # the processes to be used for training
        self.processes = []

        # queue for getting new boards
        self.board_queue = board_queue

    
    # Generate a random ship placement for use in training
    def set_board_state(self, board = np.zeros((8, 8), dtype = Config.cell_state_dtype)):
        self.board_state = board
        

    # get a randomized board state from the RBG
    def get_rbg_state(self):
        self.board_state = self.board_queue.get()
        #print("TS board state:", self.board_state)


    # Train the given Q-learning agent
    def train_agent(self, agent, table_slice):#, q_queue):
        # initialize the q-table of the agent
        #agent.init()
        # set the board for the agent
        agent.set_enemy_board(np.ndarray.flatten(table_slice))
        # train the agent
        agent.train()
        # evaluate agent
        #agent.eval()
        # once done, save the agent locally
        #agent.save_q_table()

    
    # initialize pool to prevent flooded console
    def worker_init(self):
        if Config.mute_qa:
            sys.stdout = open(os.devnull, 'w')


    # Run through the process of spinning up processes and training agents
    def run_training_processes(self, id_ = 1, iter_ = 1):
        print("TS %d subprocs %d initializing..." % (id_, iter_))
        
        # get new board
        self.get_rbg_state()

        slices = []
        # set the slices for each agent
        for i in range(int(self.num_qsteps / 2)):
            for j in range(int(self.num_qsteps / 2)):
                slices.append(self.board_state[i*self.step_size : self.step_size+i*self.step_size, j*self.step_size : self.step_size+j*self.step_size])

        # reset processes list
        self.processes = []
        # initializing the training processes
        for i in range(len(self.agents)):
            # append new process to processes list and start them
            self.processes.append(mp.Process(target = self.train_agent, args = (self.agents[i], slices[i])))
            self.processes[i].start()

        # wait for the processes to finish
        for proc in self.processes:
            proc.join()

        #args = []
        #for i in range(len(self.agents)):
        #    args.append([self.agents[i], slices[i]])

        #with mp.Pool(Config.chunk_size, initializer = self.worker_init) as pool:
        #    #pool.starmap_async(func = self.train_agent, iterable = args, chunksize = Config.chunk_size)
        #    for i in range(len(self.agents)):
        #        pool.apply_async(func = self.train_agent, args = [self.agents[i], slices[i]])
        #    pool.close()
        #    pool.join()

        end_msg = "TS %d subprocs %d finished." % (id_, iter_)
        print(end_msg)
        return end_msg


    # main process for use spawning sub processes (which then spawn subprocesses themselves)
    def run_main(self, id_ = 1):
        # go through the desired number of iterations
        print("TS %d: Beginning %d iterations." % (id_, self.num_iterations))
        
        #try:
            # create a pool to handle the iterations
        #    with mp.Pool() as pool:
                # prep the args
                # execute tasks and proc results
        #        pool.imap_unordered(func = self.run_training_processes, iterable = proc_order, chunksize = Config.chunk_size)
        #except Exception as e:
        #    print("run_main in TrainSubtables.py: EXCEPTION:\n\t%s" % str(e))
        
        proc_order = np.ravel(np.array([np.ones(self.num_iterations), np.arange(1, self.num_iterations + 1)])).reshape((self.num_iterations, 2), order = 'F').tolist()

        procs = []
        for i in range(self.num_iterations):
           # and each of them should be running the training of a full board
            procs.append(mp.Process(target = self.run_training_processes, args = (id_, i)))
            procs[i].start()
        for proc in procs:
            proc.join()
        
        print("TS %d: Finished running through iterations." % id_)


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
