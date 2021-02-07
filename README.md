# ML
ML from Scratch

# Polynomial Regression

## Description (main.py)
- This code does not use sklearn functions (from scratch)
- This code generate a polynomial (degree k) feature map (0 to k) using the training data (function create_poly).
- This code generate a polynomial (degree k) feature map (0 to k) plus sinusoidal function using the training data (function create_sin).
- This code fit linear model solving Ax = b (function fit)
- This code make a prediction given new inputs (function predict).

## Execution
```
python main.py
```

## Description (vscikit-learn.py)
- This version of code uses sklearn functions to valid the previous results.

## Execution
```
python vscikit-learn.py
```

# Logistic Regression

## Description (main.py)
- This code does not use sklearn functions (from scratch)
- This code fits a model using Newton's method to minimize the logistic regression loss function.
- This code make a prediction given new inputs (function predict).

## Execution
```
python main.py
```

## Description (vscikit-learn.py)
- This version of code uses sklearn functions to valid the previous results.

## Execution
```
python vscikit-learn.py
```

# Gaussian Discriminant Analysis

## Description (main.py)
- This code does not use sklearn functions (from scratch)
- Parameters associated to Multivariate Gaussian distribution are computed: mean vectors (mu) and covariance matrix (sigma).
- This code make a prediction given new inputs (function predict).

## Execution
```
python main.py
```

## Description (vscikit-learn.py)
- This version of code uses sklearn functions to valid the previous results.

## Execution
```
python vscikit-learn.py
```

# Poisson Regression

## Description (main.py)
- This code does not use sklearn functions (from scratch)
- This code implements a Generalized Linear Model with a Poisson distribution with gradient ascent.
- This code make a prediction given new inputs (function predict).

## Execution
```
python main.py
```

## Description (vscikit-learn.py)
- This version of code uses sklearn functions to valid the previous results.

## Execution
```
python vscikit-learn.py
```

# Naive Bayes Classifier 

## Description (main.py)
- This code does not use sklearn functions (from scratch)
- This code classifies (predicts) a message text is spam or not.
- This code make a prediction given new inputs (function predict_from_naive_bayes_model).

## Execution
```
python main.py
```

## Description (vscikit-learn.py)
- This version of code uses sklearn functions to valid the previous results.

## Execution
```
python vscikit-learn.py
```

# Support Vector Machines

## Description (main.py)
- This code does not use sklearn functions (from scratch)
- This code classifies (predicts) a message text is spam (1) or not (0).
- This code make a prediction given new inputs (function svm.train_and_predict_svm).

## Execution
```
python main.py
```

## Description (vscikit-learn.py)
- This version of code uses sklearn functions to valid the previous results.

## Execution
```
python vscikit-learn.py
```

# K-means

## Description (main.py)
- This code does not use sklearn functions (from scratch)
- This code uses k-Means to compress a image. Where the image is reduced to 16 colors (num_clusters = 16).
- Each pixel (RGB) is assigned to the closest centroid.

## Execution
```
python main.py
```

## Description (vscikit-learn.py)
- This version of code uses sklearn functions to valid the previous results.

## Execution
```
python vscikit-learn.py
```

# Semi Supervised Expectation Maximization

## Description (main.py)
- This code does not use sklearn functions (from scratch)
- This code implements Gaussian Mixture Model (GMM) to apply semi-supervised EM algorithm with labelled and unlabelled data.
- E-step and M-step are computed.
- The input data (or new data) is clustered.

## Execution
```
python main.py
```

## Description (vscikit-learn.py)
- This version of code uses sklearn functions to valid the previous results (Unsupervised Expectation-Maximization).

## Execution
```
python vscikit-learn.py
```

# Neural Network

## Description (main.py)
- This code does not use sklearn functions (from scratch)
- This code implements a simple neural network (NN) to classify grayscale images of handwritten digits (0 - 9) from the MNIST dataset.
- This NN has a single hidden layer (sigmoid function as activation) and cross entropy loss (softmax function for the output layer).

## Execution
```
python main.py
```

## Description (vTensorFlow.py)
- This version of code uses TensorFlow functions to valid the previous results.

## Execution
```
python vTensorFlow.py
```

# Reinforcement Learning -  The inverted pendulum

## Description (cartpole.py)
- This code does not use sklearn functions (from scratch)
- Parts of the code (cart and pole dynamics, and the state discretization) are inspired from code available at the RL repository http://all.cs.umass.edu//rlr//domains.html and the course  XCS229 Machine Learning from Stanford University.
- The cart-pole system is described in `cartpole.py`. The main simulation loop in this file calls the `simulate()` function for simulating the pole dynamics, `get_state()` for discretizing the otherwise continuous state space in discrete states, and `show_cart()` for display.
- The code presented in `functions.py` shows a estimate model for the underlying  Markov Decision Process (MDP), solving the Bellman's equations for this estimated MDP.

## Execution
```
python cartpole.py
```

# Reinforcement Learning -  Grid World environment

## Description
- This code does not use sklearn functions (from scratch)
- This code implements REINFORCE policy gradient algorithm to optimize on a stochastic policy that solves a simple gridded environment.

## Execution
```
python train.py
```