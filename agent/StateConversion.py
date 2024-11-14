# BD 2024
# This is done using the first n primes, where m is the number of possible states - 1. 
#   In the use case this script was made for, there are n = 25 cells of m = 3 possible states.
#   This script has been designed with more general use in mind.

# TODO:
# - [ ] add a hash table mapping function to avoid use of m**n / 2 numbers that will never be used
#       (i.e. to cut the q-table size down to 2/3 what it would be otherwise)


import numpy as np


# for the use case in mind, the 5x5 board has 25 cells
num_cells = 25          # aka "n"
# number of possible states per cell
num_cell_states = 3     # aka "m"
# list of prime numbers. only using the first two, since m - 1 = 3 - 1 = 2 for our use case, but more can be added for other use cases
primes_list = [2, 3]

# data type of the states being used
cell_state_dtype = "int8"   # using int8s for now, but a more optimal solution would only use as many bits as there are unique cell states


# Convert state to number
# TODO:
#   - make an alternative version that doesn't use num_cells, but rather the length of the state.
#       using num_cells for now for ease of use of the above parameters.
def stateToNum(state = np.zeros((num_cells), dtype = cell_state_dtype)):
    # used to keep track of the number that will be acquired throughout this function
    num = 0

    # there are n cells, so go through for each of them
    for i in range(1, num_cells + 1):
        # if the state of the current cell is 0, continue to the next iteration of the loop (i.e. skip below).
        if not state[i - 1]:
            continue

        # raise the current cell's state to the power of it's location in the board state, add to num.
        #   by "current cell's state", i mean the m - 1th prime where m is the state of the cell (which can have state ranging from 0 thru m - 1)
        num += primes_list[state[i - 1] - 1] ** i
        print(num)
    
    return num


# generate check order (i.e. which p_n**i is next greatest) for use in numToState
# i.e. if all states were all filled at once (which can't happen, but is a useful concept here), 
#   what order would we need to go in to properly catalogue them? that's what this function does.
def calcCheckOrder(cell_states = num_cell_states, cells = num_cells):
    # hold the order of the states
    order_size = (cell_states - 1) * cells      # number of states to check for the order
    #order = np.zeros((order_size, 2), dtype = "int8")       # holds actual powers/calculated numbers paired with the cell state which generated it
    # define order as a regular python array, so we can simply and efficiently insert into it at specific indices
    order = [[0, 0, 0]]

    # to be used for temp store of current cell state number
    cur_num = 0
    # iterator for use iterating thru order array
    it = 0
    # go until all of "order" is full
    for i in reversed(range(1, cells + 1)):
        # go through each of the possible cell states (- 1 to account for the 0 state) for this iteration
        for j in reversed(range(0, cell_states - 1)):
            # get current number
            cur_num = primes_list[j] ** i

            # insert current cell state just before the first instance of a value less than itself
            it = 0
            while (order[it][2] > cur_num):
                it += 1
            order.insert(it, [j, i, cur_num])   # [cell state, cell index, 3**cell index]

    # remove the element [0, 0]
    order.remove([0, 0, 0])

    
    return order
        
# notes (kinda outta date)
# - I think the current issue is that, for example, 2**25 << 3**24, so the 2 never gets checked,
#   thus the 1's are never reconstructed in numToState, since, for example, 3**24 / 2**23 >> 1
#   - okay; then, for all n, 2^n << 3^(n-1). in which case, we should swap the inner for the outer for loop
#       s.t. we go thru all 3s first, then all 2s.
#       - wait, no. then 3 would be done before 2**25.
#       - okay then, do it until we reach 2**25, then alternate or something?
#           - it'd be best to use an algorithm to pre-define how the alternation would go down,
#               rather than calculating it on the fly, which would be less efficient.
#           - start by on-the-fly calculating it, then see if you can put together an algorithm to 
#               determine the order for the first n primes up to the mth power.



# Convert number to state (old version, not functional)
#   Note:
#       Hypothetically, in the inner for loop below, int(cur_num / (primes_list[j] ** (i + 1))) should only ever be greater than 1
#   if a node in an earlier part of the state (e.g. if state of cell 10 is "earlier" than that of cell 23).
def numToState(num = 3):
    # used to put together the unique state pertaining to the given number
    state = np.zeros((num_cells), dtype = "int8")
    # get the order to check the exponents (e.g. 3**16 comes before 2**25)
    order = calcCheckOrder(num_cell_states, num_cells)
    print(order)

    # go thru starting from max to min, getting states
    cur_num = num       # used to re-trace through the production of the given number
    cur_chk = 0         # used to avoid redundant calculations
    for i in range(len(order)):
        # first, check if the cell is empty. if so, don't do any of the below, and just skip to the next iteration of the loop.
        #   do this by checking if (cur_num / smallest_prime**1) < 1. if not, it *should* mean that the state is empty.
        #   this also means that there's no required change to cur_num for this loop.
        if not int(cur_num / 2):    # if it's equal to 0, it's considered False
            continue

        # get the value which would represent the existence of the current cell state at the current cell
        cur_chk = order[i][2]
        # check if current possible state value is filled for highest state eval, i.e. primes_list[m - 1]
        if int(cur_num / cur_chk) == 1:
            # set the given cell of the state accordingly
            state[order[i][1] - 1] = order[i][0] + 1
            # update cur_num to account for how we've accounted for this part of the state
            cur_num -= cur_chk
            # remove all other instances of n to the same power, where n is any of the primes (so remove all [:,order[i][1],:])
            for j in range(len(order)):
                if order[j][1] == order[i][1]:
                    order.remove(order[j])
                    # adjust the iterators to account for the removal
                    j -= 1
                    i -= 1
    
    return state




# Convert number to state (old version, not functional)
#   Note:
#       Hypothetically, in the inner for loop below, int(cur_num / (primes_list[j] ** (i + 1))) should only ever be greater than 1
#   if a node in an earlier part of the state (e.g. if state of cell 10 is "earlier" than that of cell 23).
def numToStateOld(num = 3):
    # used to put together the unique state pertaining to the given number
    state = np.zeros((num_cells), dtype = "int8")

    # go thru starting from max to min, getting states
    cur_num = num       # used to re-trace through the production of the given number
    cur_chk = 0         # used to avoid redundant calculations
    for i in reversed(range(0, num_cells)):
        # first, check if the cell is empty. if so, don't do any of the below, and just skip to the next iteration of the loop.
        #   do this by checking if (cur_num / smallest_prime**1) < 1. if not, it *should* mean that the state is empty.
        #   this also means that there's no required change to cur_num for this loop.
        if not int(cur_num / 2):    # if it's equal to 0, it's considered False
            print(i)
            continue

        # go through num of possible cell states in reverse order (to put largest one first).
        for j in reversed(range(0, num_cell_states - 1)):   # - 1 since not accounting for 0 state
            # get the value which would represent the existence of the current cell state at the current cell
            cur_chk = primes_list[j] ** (i + 1)
            # check if current possible state value is filled for highest state eval, i.e. primes_list[m - 1]
            if int(cur_num / cur_chk) == 1:
                #print(j)
                # set the given cell of the state accordingly
                state[i] = j + 1
                print(state[i])
                # update cur_num to account for how we've accounted for this part of the state
                cur_num -= cur_chk

                # this should only trigger once per loop, so break out of the loop when it does
                break
    
    return state