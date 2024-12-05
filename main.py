from game import game, create_random_opponent

agent_wins = 0
monte_wins = 0
#for i in  range(100):
g = game(8,5, 'agent','random')
g.start()
#  if g.start() == 1:
#    agent_wins += 1
#  else:
#    monte_wins += 1
print("The q-agent won",agent_wins,"percent of the time.")
print("The random agent won",monte_wins,"percent of the time.")
