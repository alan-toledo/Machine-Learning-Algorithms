import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from scipy.optimize import curve_fit 

np.seterr(all='raise')
factor = 2.0


def func(x, a, b, c): 
    return a * np.sin(b * x) + c

def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    model = pd.read_csv(train_path)
    train_x, train_y = model['x'].values.reshape(-1, 1), model['y'].values.reshape(-1, 1)
    plot_x = np.linspace(-factor*np.pi, factor*np.pi, 1000).reshape(-1, 1)
    plt.figure()
    plt.scatter(train_x, train_y)
    scaler = preprocessing.StandardScaler()
    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        if sine == False:
            polyreg_scaled = make_pipeline(PolynomialFeatures(k), scaler, LinearRegression(fit_intercept=True))
            polyreg_scaled.fit(train_x, train_y)
            plot_y = polyreg_scaled.predict(plot_x)
        else:
            popt, _ = curve_fit(func, train_x.ravel(), train_y.ravel())
            plot_y = func(plot_x, *popt)
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x, plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, val_path):
    run_exp(train_path, sine=True, filename='vscikit_train_sine.png')
    run_exp(train_path, sine=False, filename='vscikit_train_poly.png')
    run_exp(val_path, sine=True, filename='vscikit_eval_sine.png')
    run_exp(val_path, sine=False, filename='vscikit_eval_poly.png')

if __name__ == '__main__':
    main(train_path='train.csv', val_path='valid.csv')
