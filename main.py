from game import game, create_random_opponent
import numpy as np
import sys,os
import time

# set seed for RNG seeding
np.random.seed(int(time.time() % 1000000))


agent_wins = 0
monte_wins = 0
num_games = 100

# send print to os.devnull, essentially turning off print statements from our view
init_stdout = sys.stdout
#sys.stdout = open(os.devnull, "w")

for i in range(num_games):
  g = game(8,5, 'agent','random', seed = np.random.randint(0, 999999999))
  winner = g.start()
  if winner == 1:
    agent_wins += 1
  else:
    monte_wins += 1
  np.random.seed(np.random.randint(0, 999999999))   # set new seed for next run

# send print back to console
sys.stdout = init_stdout

print("\nUsing empty q-tables:")
print("\tThe q-agent won {:10.2f} percent of the time.".format(100 * (agent_wins / num_games)))
print("\tThe random agent won {:10.2f} percent of the time.\n".format(100 * (monte_wins / num_games)))
print("\n\tNumber of q-agent wins:\t%d\n\tNumber of games:\t%d\n" % (agent_wins, num_games))
