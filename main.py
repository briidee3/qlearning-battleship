from game import game, create_random_opponent

#g = game(10,5, 'random','random')
#g.start()
opponent = create_random_opponent(8,4)
opponent.attack(0,0)
opponent.attack(1,1)

