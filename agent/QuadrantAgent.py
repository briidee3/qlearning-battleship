#BD 2024

# Application of Q-learning to playing of Battleship game
# Done as project for CS351 Intro to AI at UHart

# This particular class is for use setting up the Player agent(s) for 5x5 (sub-)boards.
# This is intended to be used in conjunction with a larger 10x10 board and three other
#   identical agents trained on the other quadrants.


import numpy as np


# define shot board states as integers representing one of three states
#   defined outside of class since is invariant
shot_states = {
            "empty":    0,
            "hit":      1,
            "miss":     2
        }

# Define a class for use in training a Q-Learning AI agent on one 5x5 quadrant of a 10x10 board game of Battleship
class QuadrantAgent:

    def __init__(self, shot_board = np.zeros((5, 5), dtype = "int8"), enemy_board = np.ndarray((5, 5), dtype = "bool"),
        discount_factor = 0.98, learn_rate = 0.8, turn_count_subtract_rate = 0.02, decay_const = 0.001, num_turns = 1000):

        ## Hyperparameters
        # discount factor (0<discount_factor<=1), importance of future rewards
        self.discount_factor = discount_factor  # slightly less than 1, so that quicker wins are preferred over longer ones
        # learning rate
        self.learn_rate = learn_rate
        # used to handle importance of minimizing number of turns, i.e. so quicker wins are preferred
        self.turn_count_subtract_rate = 0.02
        # set the decay constant used for determining explore vs exploit
        self.decay_const = decay_const
        # number of turns is essentially the number of timesteps
        self.num_turns = num_turns

        ## Q-table
        self.q_table = np.zeros((shot_board.shape[0], shot_board.shape[1]), dtype = "int8")

        ## Game state
        # action can only be True or False, so using booleans for that, no var needed to represent states
        # define the board used for representing shots (empty, hit, miss)
        self.shot_board = shot_board
        # represent true enemy board (true, false -- depends on whether ship occupies cell or not)
        self.enemy_board = enemy_board
        # used to keep track of the agent's score
        self.score = 0
        # used to keep track of number of hits thus far
        self.num_hits = 0
        # represent the passage of time by keeping track of turns 
        self.cur_turn = 1
        # number cells occupied by enemy ships; initialized to 0, then set by counting number of true states in the enemy board
        self.init_num_ships = 0
        for i in range(enemy_board.shape[0]):
            for j in range(enemy_board.shape[1]):
                if enemy_board[i][j]:
                    self.q_max += 1


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
    # returns q-value for given state-action pair, as gathered by a given coordinate pair
    def calc_q_val_bellman_at_cell(self, action_loc = np.zeros((2), dtype = "int8")):
        return self.calc_reward_at_cell(action_loc) + self.discount_factor * self.calc_max_q

    # calculate the immediate reward for taking an action (shot) at a given state
    # (represents "R(s, a)" in Bellman's equation
    def calc_reward_at_cell(self, action_loc = np.zeros((2), dtype = "int8")):
        # if the shot has already been taken, nothing can be gained from doing it again, so return 0
        if (self.shot_board[action_loc] != states["empty"]):
            return 0
        # otherwise, check if there's a boat there, reward is 1
        elif self.enemy_board[action_loc]:
            return 1
        # otherwise, 0
        else:
            return 0


    # calculate the new q value for a given state/action pair (as inferred by given coords for an action)
    def calc_new_q_val(self, action_loc = np.zeros((2), dtype = "int8")):
        return (1 - self.learn_rate) * (self.q_table[self.shot_board][action_loc]) + self.learn_rate * self.calc_q_val_bellman_at_cell(action_loc)

