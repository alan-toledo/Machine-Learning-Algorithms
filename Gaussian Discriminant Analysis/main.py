import numpy as np
import util

def main(train_path, valid_path, save_path, save_img):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
        save_img: Path to save plot linear decision boundary
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path)
    clf = GDA()
    clf.fit(x_train, y_train)
    x_eval, y_val = util.load_dataset(valid_path)
    predictions = clf.predict(x_eval)
    np.savetxt(save_path, predictions)
    print(clf.theta_0, clf.theta)
    util.plot(x_eval, y_val, clf.theta_0, clf.theta, save_img)


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5, theta_0=None, verbose=True):
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
        self.theta_0 = theta_0

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        condition = (y == 1.0)
        mu_1 = np.true_divide(x[condition].sum(axis = 0), len(x[condition]))
        mu_0 = np.true_divide(x[~condition].sum(axis = 0), len(x[~condition]))
        phi = len(x[condition])/len(x)
        sigma = np.true_divide((x[condition] - mu_1).transpose().dot(x[condition] - mu_1) + (x[~condition] - mu_0).transpose().dot((x[~condition] - mu_0)), len(x))
        sigma_inv = np.linalg.pinv(sigma)
        self.theta = -np.dot((mu_0 - mu_1).transpose(), sigma_inv)
        self.theta_0 =  -(-0.5*(np.dot(mu_0, sigma_inv).dot(mu_0) - np.dot(mu_1, sigma_inv).dot(mu_1)) - np.log((1.0 -phi)/phi))

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        predictions = []
        for row_x in x:
            theta_x = np.dot(self.theta.transpose(), row_x)
            sigmoid = 1.0/(1.0 + np.exp(-theta_x))
            predictions.append(sigmoid)
        return np.array(predictions)

if __name__ == '__main__':
    main(train_path='ds1_train.csv', valid_path='ds1_valid.csv', save_path='gda_pred_1.txt',  save_img='gda_pred_1.png')
    main(train_path='ds2_train.csv', valid_path='ds2_valid.csv', save_path='gda_pred_2.txt', save_img='gda_pred_2.png')
