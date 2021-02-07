import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from Grid_World import Grid_World
from functions import init_policy_data, choose_action, record_transition, accumulate_discounted_future_rewards, forward_pass
from REINFORCE import train_policy

#####################################
# PLAY AROUND WITH THESE PARAMETERS #
#####################################
"""
THE FOLLOWING AGENT AND ENVIRONMENT PARAMETERS CAN BE TWEAKED TO GET SOME
PRACTICE WITH THE REINFORCE ALGORITHM.  YOUR CODE WILL BE GRADED WITH THE
FOLLOWING PARAMETERS DEFINITIONS, WHICH SHOULD RESULT IN DECENT TRAINING.

# ENVIRONMENT
shape = (5,5) # The shape of the Grid World environment
num_actions = 4 # The number of actions available in each Grid World cell
rewards = np.zeros(shape) # The reward upon entering each Grid World cell 
rewards[0,4] = 1 # Top right corner = positive reward
rewards[4,0] = -1 # Bottom left corner = negative reward
reset_states = [(0,4), (4,0)] # Stops the episode at the reward cells
max_steps = 500 # Max number of steps allowed in a single episode

# TRAINING
num_training_batches = 10 # Number of times the parameters are tweaked during training.
learning_rate = 0.01 # The learning rate "alpha" from the REINFORCE algorithm
gamma = 0.5 # The MDP discount factor
num_episodes = 150 # The number of episodes to run before resetting the agent.
num_training_sessions = 20 # Smooths out reward curve when only discrete rewards are awarded.
"""

# ENVIRONMENT
shape = (5,5) # The shape of the Grid World environment
num_actions = 4 # The number of actions available in each Grid World cell
rewards = np.zeros(shape) # The reward upon entering each Grid World cell 
rewards[0,4] = 1 # Top right corner = positive reward
rewards[4,0] = -1 # Bottom left corner = negative reward
reset_states = [(0,4), (4,0)] # Stops the episode at the reward cells
max_steps = 500 # Max number of steps allowed in a single episode

# TRAINING
num_training_batches = 10 # Number of times the parameters are tweaked during training.
learning_rate = 0.01 # The learning rate "alpha" from the REINFORCE algorithm
gamma = 0.5 # The MDP discount factor
num_episodes = 150 # The number of episodes to run before resetting the agent.
num_training_sessions = 20 # Smooths out reward curve when only discrete rewards are awarded.

def convert_state_one_hot(state):
  """ Converts the (row,col) state into a multi-hot array with a bias dimension.
  """
  S = np.zeros(shape[0]+shape[1]+1)
  S[0] = 1
  S[1 + state[0]] = 1
  S[1 + shape[0] + state[1]] = 1
  return S

# Create the data structures for tracking training progress
tracked = {
  'steps' : np.zeros((num_training_sessions, num_episodes)),
  'rewards' : np.zeros((num_training_sessions, num_episodes)),
  'V' : [],
  'pi' : []
}

def record_episode(tracked, session_index, episode_index, num_steps, total_reward, policy_data, env):
  """ Records an episode for later visualization.
  
  Args:
    tracked:  dict.  Contains the tracking data structures.
    session_index:  int.
    episode_index:  int.
    num_steps:  int.  Number of steps in the episode.  Recorded parameter.
    total_reward:  float.  Total reward accumulated in the episode.  Recorded
      parameter.
  
  Returns: Nothing
  """
  tracked['steps'][session_index, episode_index] = num_steps
  tracked['rewards'][session_index, episode_index] = total_reward
  
  V = rewards.copy()
  # Only show discounted rewards if the accumulate_discounted_future_rewards()
  #   function has been implemented (does not test for correctness)
  if accumulate_discounted_future_rewards(np.array([]),0.0) is not None:
    S_hist, _, G_hist = [np.array(x) for x in zip(*policy_data['history'])]
    for state in env.all_states:
      G = V[state[0], state[1]]
      if np.any(np.all(S_hist==convert_state_one_hot(state), axis=1)) > 0:
        G = G_hist[np.all(S_hist==convert_state_one_hot(state), axis=1)].mean()
      V[state[0], state[1]] = G
  tracked['V'].append(V)

  tracked['pi'].append(forward_pass(policy_data['W'],
                                    [convert_state_one_hot(s) for s in env.all_states]))
  
  # Reset the policy_data history
  policy_data['history'] = []

def plot_stats(tracked):
  """  Plot the training and environment visualizaitons.
  
  Args:
    tracked:  dict.  Contains the tracking data structures.
    history:  
  
  Returns:  Nothing.
  """
  fig = plt.figure(figsize=(10,20))
  spec = gridspec.GridSpec(ncols=2, nrows=4, figure=fig)
  ax1 = fig.add_subplot(spec[0, :])
  ax2 = fig.add_subplot(spec[1, :])
  ax3 = fig.add_subplot(spec[2:4, :])
  # Create a plot of the number of steps required for each episode.
  ax1.plot(tracked['steps'].mean(axis=0), label='steps')
  ax1.set_ylim(bottom=0)
  ax1.set_title('Steps')
  ax1.set_xlabel('Episode')
  ax1.set_ylabel('Average Number of Steps')

  # Create a plot of the total reward accumulated over each episode
  ax2.plot(tracked['rewards'].mean(axis=0), label='rewards')
  ax2.set_title('Rewards')
  ax2.set_ylim(-1.1,1.1)
  ax2.set_xlabel('Episode')
  ax2.set_ylabel('Average Reward')

  # Create a heatmap of the average accumulated value for each state
  V = np.mean(tracked['V'], axis=0)
  img = ax3.imshow(V)
  fig.colorbar(img)

  # Create a quiver plot to show where the policy's action probability
  #   distributions
  pi_heat_map = np.mean(tracked['pi'], axis=0)
  right = np.pad(pi_heat_map[:,0].reshape(shape),0, 'constant')
  up = np.pad(pi_heat_map[:,1].reshape(shape),0, 'constant')
  left = np.pad(-pi_heat_map[:,2].reshape(shape),0, 'constant')
  down = np.pad(-pi_heat_map[:,3].reshape(shape),0, 'constant')
  kwargs = {'scale':1.8,
            'scale_units':'xy',
            'headwidth':3.,
            'headlength':3.,
            'headaxislength':3,
            'width':0.006}
  ax3.quiver(left, np.zeros(left.shape), **kwargs)
  ax3.quiver(np.zeros(down.shape), down, **kwargs)
  ax3.quiver(right, np.zeros(right.shape), **kwargs)
  ax3.quiver(np.zeros(up.shape), up, **kwargs)
  ax3.set_title(
"""Discounted Future Values (color)
Action Probability Distributions (arrows)
(only works if accumulate_discounted_future_rewards() and forward_pass() are correct)
""")
  fig.tight_layout(pad=2)
  fig.savefig('./policy_gradient_results.pdf')  

def main():
  #np.random.seed(1)
  # Create a phresh Grid World environment.
  env = Grid_World(shape=shape,
                   rewards=rewards)
  # Train multiple policies to smooth out random variations in the policies.
  for session_index in range(num_training_sessions):
    print(f'Training agent {session_index+1} of {num_training_sessions}: [', end='')
    sys.stdout.flush()
    # Initialize a new policy.
    policy_data = init_policy_data(shape[0]+shape[1], num_actions)
    # Run the specified number of episodes to train the agent.
    for episode_index in range(num_episodes):
      if episode_index % 2 == 0:
        print(f'.', end='')
        sys.stdout.flush()
      # Reset the agent's position within the environment.
      env.reset()
      # Statistics for later visualization.
      num_steps = 0
      total_rewards = 0
      # Run the episode for no more than the specified maximum number of steps.
      for _ in range(max_steps):
        # Choose an action
        action = choose_action(convert_state_one_hot(env.observation),
                               policy_data,
                               num_actions)
        # Collect the prior observation of the world.
        prior = env.observation
        # Take the step and collect the reward for that step.
        reward = env.step(action)
        # Collect the posterior observation of the world.
        posterior = env.observation
        # Record the transition for later training.
        record_transition(policy_data,
                          convert_state_one_hot(prior),
                          action,
                          reward,
                          convert_state_one_hot(posterior))
        # Statistics for later visualization.
        num_steps += 1
        total_rewards += reward
        # Stop the episode if the termination state has been reached.
        if posterior in reset_states:
          break
      # Train the policy after every episode.
      train_policy(policy_data, learning_rate, gamma, num_training_batches)
      # Record the episode statistics for later visualization.
      record_episode(tracked, session_index, episode_index, num_steps, total_rewards, policy_data, env)
    print('] Complete!')
  print('Exporting plot to policy_gradient_results.pdf!')
  plot_stats(tracked)
  
if __name__ == '__main__':
    main()
