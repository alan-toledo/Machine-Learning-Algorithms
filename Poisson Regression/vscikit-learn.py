import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
import util

def main(lr, train_path, eval_path, save_path, save_img):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    train = pd.read_csv(train_path)
    x_train, y_train = train[['x_1', 'x_2', 'x_3', 'x_4']], train[['y']].values.ravel()
    glm = PoissonRegressor(tol = 1e-5, max_iter = 10000000)
    glm.fit(x_train, y_train)

    valid = pd.read_csv(eval_path)
    x_eval, y_eval = valid[['x_1', 'x_2', 'x_3', 'x_4']], valid[['y']].values.ravel()
    predictions = glm.predict(x_eval)
    
    np.savetxt(save_path, predictions)
    util.scatter(y_eval, predictions, save_img)
    print(glm.coef_)
    print(glm.score(x_eval, y_eval))


if __name__ == '__main__':
    main(lr=1e-5, train_path='train.csv', eval_path='valid.csv', save_path='vscikit-learn_poisson_pred_valid.txt', save_img='vscikit-learn_poisson_pred_valid.png')
    main(lr=1e-5, train_path='train.csv', eval_path='test.csv', save_path='vscikit-learn_poisson_pred_test.txt', save_img='vscikit-learn_poisson_pred_test.png')