from game import game, create_random_opponent
import sys,os

agent_wins = 0
monte_wins = 0
num_games = 1000

# send print to os.devnull, essentially turning off print statements from our view
init_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")

for i in range(num_games):
  g = game(8,5, 'agent','random')
  winner = g.start()
  if winner == 1:
    agent_wins += 1
  else:
    monte_wins += 1

# send print back to console
sys.stdout = init_stdout

print("\nUsing empty q-tables:")
print("\tThe q-agent won",(agent_wins / num_games) % 0.01,"percent of the time.")
print("\tThe random agent won",(monte_wins / num_games) % 0.01,"percent of the time.\n")
print("\n\tNumber of q-agent wins:\t%d\n\tNumber of games:\t%d" % (agent_wins, num_games))
