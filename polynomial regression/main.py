import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')
factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        #(X.TX)x = X.Ty
        a = np.dot(X.transpose(), X)
        b = np.dot(X.transpose(), y)
        self.theta = np.linalg.solve(a, b)

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        polynomial = X[:, 0]
        for power in range(1 , k + 1):
            polynomial = np.column_stack((polynomial, np.power(X[:,1], power)))
        return polynomial

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        polynomial = X[:, 0]
        for power in range(1 , k + 1):
            polynomial = np.column_stack((polynomial, np.power(X[:,1], power)))
        polynomial = np.column_stack((polynomial, np.sin(X[:,1])))
        return polynomial

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        predictions = np.dot(X, self.theta)
        return predictions


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x, train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)
    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        linearModel = LinearModel()
        if sine == True:
            polynomial = linearModel.create_sin(k, train_x)
        else:
            polynomial = linearModel.create_poly(k, train_x)
        linearModel.fit(polynomial, train_y)
        #Predictions
        if sine == True:
            polynomial = linearModel.create_sin(k, plot_x)
        else:
            polynomial = linearModel.create_poly(k, plot_x)
        plot_y = linearModel.predict(polynomial)
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, val_path):
    run_exp(train_path, sine=True, filename='train_sine.png')
    run_exp(train_path, sine=False, filename='train_poly.png')
    run_exp(val_path, sine=True, filename='eval_sine.png')
    run_exp(val_path, sine=False, filename='eval_poly.png')

if __name__ == '__main__':
    main(train_path='train.csv', val_path='valid.csv')
