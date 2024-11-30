from player import player
from ship import ship

def create_random_opponent(board_size, num_ships):
  random_opponent = player(board_size, 'random')
  ships = [2, 3, 3, 4, 5]
  ship_num = 0
  while ship_num < num_ships:
    placement = random_opponent.ship_input(ships[ship_num])
    row = placement[0]
    col = placement[1]
    direction = placement[2]
    if random_opponent.place_ship_without_printing(row, col, ships[ship_num], direction) == 6:
      ship_num = ship_num + 1
  random_opponent.print_board()
  return random_opponent

class game:
  def __init__(self, board_size, num_ships, player1_type, player2_type):
    self.num_ships = num_ships
    self.board_size = board_size
    carrier = ship(5)
    battleship = ship(4)
    cruiser = ship(3)
    submarine = ship(3)
    destroyer = ship(2)
    self.ships = [destroyer, submarine, cruiser, battleship, carrier]
    self.player1 = player(board_size, player1_type)
    self.player2 = player(board_size, player2_type)

  def start(self): #let the games begin
    self.place_phase()
    print("Player", self.shooting_phase(),"wins!")

  def place_phase(self): #asks for ship placements from player 1, then player 2
    print("Player 1 placing phase:")
    ship = 0
    while ship < self.num_ships:
      placement = self.player1.ship_input(self.ships[ship].get_length())
      row = placement[0]
      col = placement[1]
      direction = placement[2]
      if self.player1.place_ship(row, col, self.ships[ship], direction) != 6:
        print("Try again")
      else:
        self.player1.print_board()
        ship = ship + 1
    print("Player 2 placing phase:")
    ship = 0
    while ship < self.num_ships:
      placement = self.player2.ship_input(self.ships[ship].get_length())
      row = placement[0]
      col = placement[1]
      direction = placement[2]
      if self.player2.place_ship(row, col, self.ships[ship], direction) != 6:
        print("Try again")
      else:
        self.player2.print_board()
        ship = ship + 1

  def shooting_phase(self): #asks for player 1's guesses and player 2's guesses back and forth until someone wins
    winner = 0
    win_condition = 0
    for i in range(self.num_ships):
      win_condition += self.ships[i].get_length()
    while winner == 0:
      success = False
      while not success:
        position = self.player1.shoot_input(self.player2.enemy_guesses,self.player2)
        row = position[0]
        col = position[1]
        if self.player2.attack(row, col) == -1:
          print("Try again")
        else:
          self.player2.print_guesses()
          score = self.player2.update_score()
          print("Score:",score)
          success = True
          if score == win_condition:
            return 1 #returns the winner as player 1
      success = False
      while not success:
        position = self.player2.shoot_input(self.player1.enemy_guesses,self.player1)
        row = position[0]
        col = position[1]
        if self.player1.attack(row, col) == -1:
          print("Try again")
        else:
          self.player1.print_guesses()
          score = self.player1.update_score()
          print("Score:", score)
          success = True
          if score == win_condition:
            return 2 #returns the winner as player 2
    return winner
