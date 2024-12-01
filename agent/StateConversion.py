# BD 2024
# This is done using the first n primes, where m is the number of possible states - 1. 
#   In the use case this script was made for, there are n = 25 cells of m = 3 possible states.
#   This script has been designed with more general use in mind.

# TODO:
# - [ ] add a hash table mapping function to avoid use of m**n / 2 numbers that will never be used
#       (i.e. to cut the q-table size down to 2/3 what it would be otherwise)
# NOTE: This is only intended for use in an implementation in which the agent is expected to output
#   a boolean action. The cells may represent more than 2 states, but this is essentially creating and
#   using a boolean decision tree, since either a shot is taken or it is not (i.e. application to Battleship board game).
# NOTE: This version of StateConversion.py takes the sum of primes representing the cells on the board for
#   each possible cell state and pairs them using a deterministic pairing function. This one may or may not be
#   utilized in the final version of the program.

import numpy as np
from . import Config


# Convert state to number
#   essentially just base conversion (see the blurb above "num_to_state" for more info)
def state_to_num(state = np.zeros((Config.num_cells), dtype = Config.cell_state_dtype), base = Config.num_cell_states, cell_count = Config.num_cells):
    # used to keep track of the number that will be acquired throughout this function
    num = 0

    # there are n cells, so go through for each of them
    for i in range(0, cell_count):
        # if the state of the current cell is 0, continue to the next iteration of the loop (i.e. skip below).
        if not state[i]:
            continue

        # otherwise, add the current power times the cell's state
        num += int(state[i]) * (base ** i)
    
    return num


# The problem of converting num to state is actually much simpler than I was making it; all it really is is converting a base 10
#   number to base <num_cell_states>, where each of the digits in the base <num_cell_states> representation are a cell state.
#   https://stackoverflow.com/questions/38448013/how-can-i-list-all-possibilities-of-a-3x3-board-with-3-different-states
# TL;DR: This function is essentially just converting the given number (num) to base <num_cell_states> (e.g. base 3 for our purposes)
def num_to_state(num = int(3), base = Config.num_cell_states, cell_count = Config.num_cells):
    # return 0 when num is empty
    if num == 0:
        return np.zeros(cell_count, dtype = Config.cell_state_dtype)
    
    # initialize the list to store non-empty cells (and empty cells in between)
    digits = []

    # keep going until num < 1, i.e. until it can only be 0 when cast to int
    while num >= 1:
        # get the next base <num_cell_states> digit to be output
        digits.append(int(num % base))
        num = int(num / base)
    
    # put the new base <num_cell_states> number into an array to represent the board state
    return np.array(([0 for _ in range(cell_count - 1)] + digits[::-1])[-cell_count:][::-1], dtype = Config.cell_state_dtype)


