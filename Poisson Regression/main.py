import numpy as np
import util
import matplotlib.pyplot as plt
import pdb

def main(lr, train_path, eval_path, save_path, save_img):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path)

    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train, y_train)
    
    x_eval, y_eval = util.load_dataset(eval_path)
    predictions = clf.predict(x_eval)

    np.savetxt(save_path, predictions)

    util.scatter(y_eval, predictions, save_img)
    print(clf.theta)

class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,  theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])
        
        for _ in range(self.max_iter):
            last_theta = self.theta
            
            poisson_regression = np.zeros(x.shape[1])
            for row_x, value_y in zip(x, y):
                theta_dot_x = np.dot(self.theta.transpose(), row_x)
                poisson_regression = poisson_regression + (float(value_y) - np.exp(theta_dot_x))*row_x
            
            self.theta = self.theta + self.step_size*poisson_regression
            
            diff = np.linalg.norm((self.theta - last_theta), ord=1)
            if diff < self.eps:
                break

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        predictions = []
        for row_x in x:
            theta_dot_x = np.dot(self.theta.transpose(), row_x)
            predictions.append(np.exp(theta_dot_x))
        return np.array(predictions)

if __name__ == '__main__':
    main(lr=1e-5, train_path='train.csv', eval_path='valid.csv', save_path='poisson_pred_valid.txt', save_img='poisson_pred_valid.png')
    main(lr=1e-5, train_path='train.csv', eval_path='test.csv', save_path='poisson_pred_test.txt', save_img='poisson_pred_test.png')
