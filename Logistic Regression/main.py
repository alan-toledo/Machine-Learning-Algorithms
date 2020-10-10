import numpy as np
import util
import pdb

def main(train_path, valid_path, save_path, save_img):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
        save_img: Path to save plot classification.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    x_eval, y_val = util.load_dataset(valid_path, add_intercept=True)
    predictions = clf.predict(x_eval)
    np.savetxt(save_path, predictions)
    print(clf.theta)
    util.plot(x_eval, y_val, clf.theta, save_img)


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5, theta_0=None, verbose=True):
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])
        
        for _ in range(self.max_iter):
            temp_theta = self.theta

            log_likelihood = np.zeros(x.shape[1])
            diagonal = []
            for row_x, value_y in zip(x, y):
                gthetaX = 1.0/(1.0 + np.exp(-np.dot(self.theta.transpose(), row_x)))
                log_likelihood = log_likelihood + ((value_y -  gthetaX)*row_x)/x.shape[0]
                diagonal.append(gthetaX*(1.0 - gthetaX)/x.shape[0])
            D = np.diag(diagonal)
            H = np.dot(x.transpose(), D)
            #Hessian
            H = np.dot(H, x)
            #Inverse of The Hessian
            H_inverse = np.linalg.pinv(H)
            #Run Newton's Method to minimize J(theta)
            self.theta = self.theta - self.step_size*np.dot(H_inverse, -log_likelihood)

            diff = np.linalg.norm((self.theta - temp_theta), ord=1)
            #Threshold for determining convergence.
            if diff < self.eps:
                break


    def predict(self, x):
        """Return predicted probabilities given new inputs x.

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
    main(train_path='ds1_train.csv', valid_path='ds1_valid.csv', save_path='logreg_pred_1.txt', save_img='logreg_pred_1.png')
    main(train_path='ds2_train.csv', valid_path='ds2_valid.csv', save_path='logreg_pred_2.txt', save_img='logreg_pred_2.png')
