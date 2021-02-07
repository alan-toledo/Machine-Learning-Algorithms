# Reinforcement Learning -  The inverted pendulum

## Description (cartpole.py)
- This code does not use sklearn functions (from scratch)
- Parts of the code (cart and pole dynamics, and the state discretization) are inspired from code available at the RL repository http://www-anw.cs.umass.edu/rlr/domains.html and the course  XCS229 Machine Learning from Stanford University.
- The cart-pole system is described in `cartpole.py`. The main simulation loop in this file calls the `simulate()` function for simulating the pole dynamics, `get_state()` for discretizing the otherwise continuous state space in discrete states, and `show_cart()` for display.
- The code presented in `functions.py` shows a estimate model for the underlying  Markov Decision Process (MDP), solving the Bellman's equations for this estimated MDP.

## Execution
```
python cartpole.py
```
