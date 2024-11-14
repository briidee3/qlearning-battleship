import numpy as np
from numpy import ndarray

class ship:
  ship: ndarray

  def __init__(self, length):
    self.length = length
    self.ship = np.ndarray(length, dtype=bool)

  def get_status(self,i):
    return self.ship[i]

  def is_sunk(self):
    sunk = True
    for i in range(len(self.ship)):
      sunk &= not self.ship[i]
    return sunk

  def get_length(self):
    return self.length

