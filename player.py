from ship import ship
class player:
  def __init__(self, board_size):
    self.enemy_score = 0
    self.board_size = board_size
    # Initialize a board with empty cells
    self.board = [['~' for _ in range(board_size)] for _ in range(board_size)]
    self.enemy_guesses = [[2 for _ in range(board_size)] for _ in range(board_size)]

  def print_board(self):
    # Print the board in a readable format
    print("  " + " ".join(map(str, range(self.board_size))))
    for i, row in enumerate(self.board):
      print(f"{i} " + " ".join(row))
    print()

  def print_guesses(self):
    # Print the board in a readable format
    print("  " + " ".join(map(str, range(self.board_size))))
    for i, row in enumerate(self.enemy_guesses):
      print(f"{i} " + " ".join(map(str,row)))
    print()

  def place_ship(self, row, col, ship, direction):
    """
    Places a ship on the board.

    Parameters:
    - row, col: Starting position of the ship
    - length: Length of the ship
    - direction: 'H' for horizontal or 'V' for vertical
    """
    if direction == 'H':
      if col + ship.get_length() > self.board_size:
        print("Ship placement goes out of bounds horizontally.")
        return 1
      for i in range(ship.get_length()):
        if self.board[row][col + i] != '~':
          print("Space is already occupied by another ship.")
          return 2
      for i in range(ship.get_length()):
        self.board[row][col + i] = 'S'
      return 6

    elif direction == 'V':
      if row + ship.get_length() > self.board_size:
        print("Ship placement goes out of bounds vertically.")
        return 3
      for i in range(ship.get_length()):
        if self.board[row + i][col] != '~':
          print("Space is already occupied by another ship.")
          return 4
      for i in range(ship.get_length()):
        self.board[row + i][col] = 'S'
      return 6

    else:
      print("Invalid direction. Use 'H' for horizontal or 'V' for vertical.")
      return 5

  def attack(self, row, col):
    """
    Mark an attack on the board.

    Parameters:
    - row, col: Coordinates to attack

    Returns:
    - 'Hit' if a ship is there, 'Miss' otherwise.
    """
    if self.enemy_guesses[row][col] != 2:
      print("Already attacked here")
      return -1
    elif self.board[row][col] == 'S':
      self.enemy_guesses[row][col] = 0
      print("Hit")
      return 1
    elif self.board[row][col] == '~':
      self.enemy_guesses[row][col] = 1
      print("Miss")
      return 0

  def update_score(self):
    score = 0
    for row in self.enemy_guesses:
      score += row.count(0)
    return score

  def win_condition(self):
    win_condition = 0
    for row in self.board:
      win_condition += row.count('s')
    return win_condition
