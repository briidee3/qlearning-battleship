from player import player
from ship import ship

class game:
  def __init__(self, board_size, num_ships):
    self.num_ships = num_ships
    self.board_size = board_size
    carrier = ship(5)
    battleship = ship(4)
    cruiser = ship(3)
    submarine = ship(3)
    destroyer = ship(2)
    self.ships = [destroyer, submarine, cruiser, battleship, carrier]
    self.player1 = player(board_size)
    self.player2 = player(board_size)

  def start(self): #let the games begin
    self.place_phase()
    print("Player", self.shooting_phase(),"wins!")

  def place_phase(self): #asks for ship placements from player 1, then player 2
    print("Player 1 placing phase:")
    ship = 0
    while ship < self.num_ships:
      print("Where does player 1 want to place the ship of length",self.ships[ship].get_length(),"?")
      print("Enter row 0 to",self.board_size-1,":")
      row = int(input())
      print("Enter column 0 to",self.board_size-1,":")
      col = int(input())
      print("Enter direction (H or V) :")
      direction = input()
      if self.player1.place_ship(row, col, self.ships[ship], direction) != 6:
        print("Try again")
      else:
        self.player1.print_board()
        ship = ship + 1
    print("Player 2 placing phase:")
    ship = 0
    while ship < self.num_ships:
      print("Where does player 2 want to place the ship of length",self.ships[ship].get_length(),"?")
      print("Enter row 0 to",self.board_size-1,":")
      row = int(input())
      print("Enter column 0 to",self.board_size-1,":")
      col = int(input())
      print("Enter direction (H or V) :")
      direction = input()
      if self.player2.place_ship(row, col, self.ships[ship], direction) != 6:
        print("Try again")
      else:
        self.player2.print_board()
        ship = ship + 1

  def shooting_phase(self): #asks for player 1's guesses and player 2's guesses back and forth until someone wins
    winner = 0
    win_condition = 17 #for a full size board
    while winner == 0:
      success = False
      while not success:
        print("Player 1's guesses (2:unknown, 1:miss, 0:hit):")
        self.player2.print_guesses()
        print("Where does player 1 want to attack ?")
        print("Enter row 0 to", self.board_size - 1, ":")
        row = int(input())
        print("Enter column 0 to", self.board_size - 1, ":")
        col = int(input())
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
        print("Player 2's guesses (2:unknown, 1:miss, 0:hit):")
        self.player1.print_guesses()
        print("Where does player 2 want to attack ?")
        print("Enter row 0 to", self.board_size - 1, ":")
        row = int(input())
        print("Enter column 0 to", self.board_size - 1, ":")
        col = int(input())
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
