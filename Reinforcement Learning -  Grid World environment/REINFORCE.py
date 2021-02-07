import numpy as np
from functions import accumulate_discounted_future_rewards, policy_gradient


def train_policy(policy_data, learning_rate, gamma, num_epochs):
  """  Performs the REINFORCE policy gradient training.
  
  Args:
    policy_data:  dict.  The policy parameters and tracking structures that you
      intialized in init_policy_data()
    learning_rate: float.  Equivalent to "alpha" in the REINFORCE algorithm.
    gamma:  float.  The MDP discount factor.
    num_epochs:  The number of times to train the policy weights.
      
  Returns:  Nothing.
  """
  if policy_data['episode']:
    
    # Convert the episode's transitions into numpy arrays, then clear episode data
    S, A, R = [np.array(x) for x in zip(*policy_data['episode'])]
    policy_data['episode'] = []

    # Calculate the discounted sum of future rewards
    G = accumulate_discounted_future_rewards(R, gamma)
    
    # Save the transitions for training visualization
    policy_data['history'].extend(zip(S, A, G))

    # Perform the policy gradient training
    for _ in range(num_epochs):
      diff_W = policy_gradient(policy_data['W'], S, A, G, learning_rate)
      policy_data['W'] += diff_W
