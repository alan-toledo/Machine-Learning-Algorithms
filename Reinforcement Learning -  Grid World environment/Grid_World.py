import numpy as np

def coordinate_step(state, action, shape):
  """ Determines the obvious posterior state given an action and prior state.
  
  This adgeres to the following convention:
  - 0 (RIGHT): increase col by 1
  - 1 (UP): decrease row by 1
  - 2 (LEFT): decrease col by 1
  - 3 (DOWN): increase row by 1
  - Attempting to leave the grid or cross a barrier will not move the agent.

  Args:
    state:  tuple. (row, col).  Contains 2 ints indicating the prior position of
      the agent in the Grid World.
    action:  int. In the set [0,3].  Indicates the action (see above).
    shape:  tuple. (num_rows, num_cols).  Contains 2 ints indicating the shape
      of the Grid World.
      
  Returns:
    posterior_state:  tuple. (row, col).  Contains 2 ints indicating the
      posterior position of the agent in the Grid World.
  """
  row, column = state
  if action == 1:
    row -= 1
    if row < 0:
      row += 1
  elif action == 0:
    column += 1
    if column >= shape[1]:
      column -= 1
  elif action == 3:
    row += 1
    if row >= shape[0]:
      row -= 1
  elif action == 2:
    column -= 1
    if column < 0:
      column += 1
  return row, column

class Grid_World:
  """ A gridded 2D environment in which to practice RL algorithms.
  
  The Grid World will maintain state that tracks an agent as it explores and
    earns rewards.  The Grid World has the following attributes:
    - The environment is a rectangular grid of cells
    - An agent can perform four actions in each cell (up, right, down, left).
    - Rewards are associated with entering a cell, regardless of action
      or departure cell.
  
  Attributes:
    _shape:  tuple. (num_rows, num_cols).  Contains 2 ints indicating the shape
      of the Grid World.
    _initial_position:  tuple. (row, col).  Contains 2 ints indicating the
      starting position of the agent in the Grid World.
    _position:  tuple. (row, col).  Contains 2 ints indicating the position of
      the agent in the Grid World.
    _R: np.ndarray, dtype=np.float64, shape=(num_rows, num_cols).  Stores the
      reward for entering each cell.
  """

  def __init__(self,
               shape=(4,4),
               initial_position=(0,0),
               rewards=None):
    """ Defines a new environment.
    
    Args:
      (see class definition)
    """
    self._shape = shape
    self._initial_position = initial_position
    self._R = rewards
  
  @property
  def observation(self):
    """ Returns the current observations of the agent in Grid World.
    
    In the base class (this one), this is simply the (row, col) of the agent.
      This method can be overidden to partially obscure part of the state and
      train a Partially Observable MDP (POMDP).
    """
    return self.state
  
  @property
  def state(self):
    """ Returns the current world-state of the agent in Grid World.
    """
    return self._position
  
  def step(self, action):
    """ Changes the agent's position in Grid World.
    
    Args:
      action:  (see coordinate_step() function)
    
    Returns:
      reward: reward for entering the posterior state
    """
    self._position = coordinate_step(self._position, action, self._shape)
    return self._R[self._position]
  
  def reset(self):
    """ Reset's the agent's position to the intial position in Grid World.
    """
    self._position = self._initial_position
  
  @property
  def all_states(self):
    """ Utility function that returns a list of all states in Grid World.
    """
    return np.array(np.meshgrid(np.arange(self._shape[0]),
                                np.arange(self._shape[0]),
                                indexing='ij')).reshape(2,-1).T
