# BD 2024
# Evaluate the agent against a monte carlo opponent

import numpy as np
import random
import multiprocessing as mp
import time

import agent.Config as cfg
import agent.TrainSubtables as ts
import agent.QAgent as qa
import agent.TablePlayer as tp
from game import create_random_opponent


ships = [2,2,3,3,4]
num_ship_cells = np.sum(ships)


# run through one game loop
def play_game():
    # keep track of number of turns
    num_turns = 0
    # keep track of scores
    scores = [0, 0]     # agent, random
    # generate a random board for each player (q-agent player 1, monte carlo player 2)
    enemy_boards = [gen_random_board(), np.ndarray.flatten(gen_random_board())]   # agent, random
    # create boards to represent the shots already taken
    shots_boards = [np.zeros((64), dtype = cfg.cell_state_dtype)] * 2
    # keep track of the targets of each player
    targets = [0, 0]    # agent, random

    # create an agent to play the game
    table_player = tp.TablePlayer()
    # set the board state for the q-agent player
    table_player.set_enemy_board_state(enemy_boards[0])
    # flatten the board after giving it to the player, for coherence with the other board below
    enemy_boards[0] = np.ndarray.flatten(enemy_boards[0])

    # run through the game until someone wins or loses
    while num_ship_cells not in scores:
        # iterate turn number
        num_turns += 1

        # represent monte carlo target index
        targets[1] = np.random.randint(64)
        # generate a random number until it gets one corresponding to an empty cell
        while 0 in shots_boards[1] and shots_boards[1][targets[1]] != 0:
            targets[1] = random.randint(0, 63)

        # get the index for the move of the agent player
        targets[0] = table_player.step()

        # update both boards accordingly
        for i in range(2):
            # check if there's a ship at the location taken to shoot at
            if enemy_boards[i][targets[i]] == 1:
                # iterate score for this player
                scores[i] += 1
                # update the shots-board to denote that a shot was taken here
                shots_boards[i][targets[i]] = 1
            # otherwise it's a miss, update accordingly
            else:
                shots_boards[i][targets[i]] = 2
    
    # declare the winner, set up statistics
    #print("Current game finished.")
    statistics = []
    # check if agent won
    if scores[0] > scores[1]:   #win
        statistics.append(1)
        print("\tWinner: Q-Learning agent.")
    else:                       #loss
        statistics.append(-1)
        print("\tWinner: Monte Carlo agent")

    # append the difference between scores to stats
    statistics.append(scores[0] - scores[1])
    # add the number of turns taken to stats
    statistics.append(num_turns)

    # return the stats for the game
    return statistics


# create random board
def gen_random_board():
    rand_board = np.ndarray.flatten(np.array(create_random_opponent(8, ships).get_board()))
    rand_board[rand_board == 'S'] = 1
    rand_board[rand_board == '~'] = 0
    rand_board = np.array(rand_board.reshape((8, 8)), dtype = cfg.cell_state_dtype)

    return rand_board


# run through the given number of games, then print out stats about them
def evaluate(num_games = 1):
    # used to calculate average score at the end
    score_sum = 0
    # used to calculate win/loss ratio for the agent at the end
    wins = 0
    # used to keep track of avg number of turns in a game
    turns_sum = 0

    # go through all of the games
    for i in range(num_games):
        print("Game %d: " % i)
        cur_stats = play_game()
        wins += cur_stats[0]
        score_sum += cur_stats[1]
        turns_sum += cur_stats[2]

    # calculate and print out stats
    print("Evaluation complete.\n\tWin/loss ratio: %d\n\tAverage score: %d\n\tAverage number of turns: %d" % 
        ((wins / num_games), (score_sum / num_games), (turns_sum / num_games)))


if __name__ == "__main__":
    evaluate(100)
