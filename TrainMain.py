# BD 2024
# Class for use running the training processes 

import numpy as np
import multiprocessing as mp
import time

import agent.Config as cfg
import agent.TrainSubtables as ts
import agent.QAgent as qa
from game import create_random_opponent


# ships to be used
ships = [2, 2, 3, 3, 4]

# test board for initial testing of the traning processes
test_board = np.array([[0, 0, 1, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0]], dtype = cfg.cell_state_dtype)

# test the training process
def test():
    print("Initializing trainer...")
    trainer = ts.TrainSubtables(step_size = 4, num_iterations = 1)
    trainer.set_board_state(test_board)
    print("Trainer initialized.")

    print("Beginning training processes...")
    trainer.run_training_processes()


# create a random board every so often
def random_board_generator(queue = mp.Queue(), pipe = mp.Pipe()[0]):
    print("RBG initialized.")
    while True:
        # always keep the board queue full, but not overfilled
        if queue.qsize() < 100:
            # create new random board
            rand_board = np.ndarray.flatten(np.array(create_random_opponent(8, ships).get_board()))
            rand_board[rand_board == 'S'] = 1
            rand_board[rand_board == '~'] = 0
            rand_board = np.array(rand_board.reshape((8, 8)), dtype = cfg.cell_state_dtype)

            # send board out to pipe
            queue.put(rand_board)
        # check for exit
        if pipe.poll():
            if pipe.recv() == "end_run":
                break
    print("RBG finished.")


# initialize TrainSubtables proc instance
def ts_init(ts, id_ = 1):
    # go through the designated amount of iterations of random board states,
    #   spawning 4 child procs (one per quadrant) of QAgent per iteration.
    ts.run_main(id_)

if __name__ == "__main__":
    #try:
        # set the method for spawning processes
    # set method for spawning process if not windows
    if os.name != "nt":
        mp.set_start_method('fork', force = True)
    #except RuntimeError:
    #    pass


    # set up queue and pipes for sending and receiving randomly generated boards
    board_queue = mp.Queue()
    sendit, halfpipe = mp.Pipe()

    # set up random board generator
    rbg_proc = mp.Process(target = random_board_generator, args = (board_queue, halfpipe))
    # start the RBG
    rbg_proc.start()


    # initialize the TrainSubtables processes
    ts_procs = []
    ts_objs = []
    for i in range(cfg.num_ts_subprocs):
        ts_objs.append(ts.TrainSubtables((8,8), 4, cfg.num_ts_iter, board_queue))
        ts_procs.append(mp.Process(target = ts_init, args = [ts_objs[i], i + 1]))
        ts_procs[i].start()

    #with mp.Pool() as pool:
    #    pool.imap_unordered(func = ts_init, iterable = ts_objs, chunksize = Config.chunk_size)


    # wait for processes to end
    for proc in ts_procs:
        proc.join()

    # end the RBG
    sendit.send("end_run")
    rbg_proc.join()
