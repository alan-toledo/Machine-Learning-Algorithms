import numpy as np
import pdb
def softmax(x):
  """
  Compute softmax function for a batch of input values. 
  The first dimension of the input corresponds to the batch size. The second dimension
  corresponds to every class in the output. When implementing softmax, you should be careful
  to only sum over the second dimension.

  Note that, in the case of the REINFORCE algorithm, the num_classes is actually
  num_actions, since we are predicting the highest value action, given the
  current state.

  Important Note: You must be careful to avoid overflow for this function. Functions
  like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
  You will know that your function is overflow resistent when it can handle input like:
  np.array([[10000, 10010, 10]]) without issues.

  Args:
    x: np.ndarray, dtype=np.float64, (batch_size, num_classes).

  Returns:
    np.ndarray, dtype=np.float64, (batch_size, num_classes).  The softmax
      of the input, x.
  """
  # *** START CODE HERE (~4 lines) ***
  tempX = x - (np.max(x, axis=1)).reshape(x.shape[0], 1)
  p = np.exp(tempX) / np.sum(np.exp(tempX), axis=1, keepdims=True)
  return p
  # *** END CODE HERE ***

def forward_pass(W, S):
  """ Calculates a probability distribution over all possible actions.
  
  Args:
    W: np.ndarray, dtype=np.float64, (num_state_params+1, num_actions). Contains
      the trainable weights defining the agent's policy.
    S: np.ndarray, dtype=np.float64, (-1, num_state_params+1).  Row-list of
      states for which to calculate the forward pass.
  
  Returns:
    action_probs:  np.ndarray, dtype=np.float64, (-1, num_actions).  Row-list of
      probability distributions over all actions.
  """
  # *** START CODE HERE (~1 line) ***
  prob = np.dot(S, W)
  prob = prob.reshape(-1, W.shape[1])
  return softmax(prob)
  # *** END CODE HERE ***

def policy_gradient(W, S, A, G, learning_rate):
  """ Calculates the weight update for the REINFORCE algorithm.
  
  Args:
    W: np.ndarray, dtype=np.float64, (num_state_params+1, num_actions). Contains
      the trainable weights defining the agent's policy.
    S:  np.ndarray, dtype=np.float64, (-1, num_state_params+1).  Row-list of
      states encountered in the episode.
    A:  np.ndarray, dtype=np.float64, (-1,).  The index of the action chosen by
      the agent.
    G:  np.ndarray, dtype=np.float64, (-1,).  The discounted sum of future
      rewards for each step in an episode.
    learning_rate: float.  Equivalent to "alpha" in the REINFORCE algorithm.

  Returns:
    diff_W: np.ndarray, dtype=np.float64, (num_state_params+1, num_actions).
      The REINFORCE weight updates for W.
  """
  # *** START CODE HERE (~6 lines) ***
  prob = forward_pass(W, S) # (-1, num_actions)
  grad = np.zeros(prob.shape)
  for i in range(prob.shape[0]):
    for j in range(prob.shape[1]):
      if A[i] == j:
        grad[i, j] = G[i]*(1.0 - prob[i, j])
      else:
        grad[i, j] = -G[i]*prob[i, j]
  diff_W = learning_rate*np.dot(S.T, grad)
  return diff_W
  # *** END CODE HERE ***

def init_policy_data(num_state_params, num_actions):
  """ Initialize the policy's weights and structures for tracking training data.
  
  The policy's weight parameters should be initialized by choosing from a random
    normal distribution (c.f. np.random.normal()).  The episode will be tracked
    using a simple numpy array.
  
  Args:
    num_state_params:  The number of features in the state (no including the
      bias term)
    num_actions:  The number of actions for the environment
    
  Returns:
    dict. Two items:
      'W' : np.ndarray, dtype=np.float64, (num_state_params+1,num_actions)
      'episode' : data structure for tracking the episode's transitions.  Will
        be used for training after the episode is complete.
      'history' : data structure for tracking prior episodes (used for
        visualizations only).
  """
  W = None
  # *** START CODE HERE (~1 lines) ***
  W = np.random.normal(size = (num_state_params + 1,num_actions))
  # *** END CODE HERE ***

  return {'W' : W,
          'episode' : [],
          'history' : []}

def choose_action(state, policy_data, num_actions):
  """ Chooses an action from the policy's action probability distribution.
  
  Hint:  Remember that you can use the forward_pass() function to calculate the
    policy's action probability distribution.
  
  Args:
    state:  np.ndarray, dtype=np.float64, (num_state_params+1,).  The multi-hot
      vector representation of the current state.
    policy_data:  dict.  The policy parameters and tracking structures that you
      intialized in init_policy_data()
    num_actions:  
  
  Returns:
    int.  The index of the chosen action.
  """
  # *** START CODE HERE (~2 lines) ***
  action_probs = forward_pass(policy_data['W'], state)[0]
  return np.random.choice(num_actions, p=action_probs)
  # *** END CODE HERE ***
  
def record_transition(policy_data, prior_state, action, reward, posterior_state):
  """ Records the transition for later training.
  
  Record the transition in your policy_data episode tracking data structure.
    You may not need all of the data provided to this function (s, a, r, s')
  
  Args:
    policy_data:  dict.  The policy parameters and tracking structures that you
      intialized in init_policy_data()
    prior_state:  np.ndarray, dtype=np.float64, (num_state_params+1,).  The
      multi-hot vector representation of the state prior to the transition.
    action:  int.  The index of the action causing the transition.
    reward:  float.  The reward resulting from the transition.
    posterior_state:  np.ndarray, dtype=np.float64, (num_state_params+1,).  The
    multi-hot vector representation of the state after the transition.
    
  Returns:  Nothing.
  """
  # *** START CODE HERE (~1 line) ***
  policy_data['episode'].append([posterior_state, action, reward])
  # *** END CODE HERE ***

def accumulate_discounted_future_rewards(R, gamma):
  """  Calculates the discounted sum of future rewards for each episode step.
  
  Args:
    R:  np.ndarray, dtype=np.float64, (episode_length,)
    gamma:  float.  The MDP discount factor.
    
  Returns:
    np.ndarray, np.float64, (episode_length,).  G, the discounted sum of future
      rewards for each step in the episode.
  """
  # *** START CODE HERE (~4 lines) ***
  G = np.zeros(len(R))
  for i in range(len(R)):
    temp = 0.0
    for j in range(i, len(R)):
      temp =  temp + np.power(gamma, j - i)*R[j]
    G[i] = temp
  return G
  # *** END CODE HERE ***
