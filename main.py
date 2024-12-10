from game import game, create_random_opponent
import numpy as np
import sys,os
import time

# set seed for RNG seeding
np.random.seed(int(time.time() % 1000000))


agent_wins = 0
monte_wins = 0
num_games = 100
turns_taken = [0, 0]

# send print to os.devnull, essentially turning off print statements from our view
init_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")

for i in range(num_games):
  g = game(8,5, 'agent','random', seed = np.random.randint(0, 99999999))
  winner, cur_turns_taken = g.start()
  if winner == 1:
    agent_wins += 1
    turns_taken[0] += cur_turns_taken
  else:
    monte_wins += 1
    turns_taken[1] += cur_turns_taken
  np.random.seed(np.random.randint(0, 999999999))   # set new seed for next run
  #print(g.turn)
  #print(g.player1.table_player.num_shots)
  #print(g.player1.table_player.num_hits)

# send print back to console
sys.stdout = init_stdout

print("\nUsing trained q-tables:")
print("\tThe q-agent won {:10.2f} percent of the time.".format(100 * (agent_wins / num_games)))
print("\tThe random agent won {:10.2f} percent of the time.\n".format(100 * (monte_wins / num_games)))
print("\n\tNumber of q-agent wins:\t%d\n\tNumber of games:\t%d\n" % (agent_wins, num_games))
print("\nAvg number of turns:\n\tQ-agent: {:10.2f}\n\tRandom: {:10.2f}".format((turns_taken[0] / agent_wins), (turns_taken[1] / monte_wins)))
print("Num wins p1: %d\nNum turns p1: %d" % (agent_wins, turns_taken[0]))
print("Num wins p2: %d\nNum turns p2: %d" % (monte_wins, turns_taken[1]))