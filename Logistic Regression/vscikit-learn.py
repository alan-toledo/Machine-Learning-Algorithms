import numpy as np
import pandas as pd
import util
from sklearn.linear_model import LogisticRegression

def main(train_path, valid_path, save_path, save_img):
    """Problem: Logistic regression version sklearn.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted using np.savetxt().
        save_img: Path to save plot classification.
    """
    train = pd.read_csv(train_path)
    train['intercept'] = 1.0
    x_train, y_train = train[['intercept', 'x_1', 'x_2']], train[['y']].values.ravel()
    clf = LogisticRegression(random_state=0, fit_intercept=False).fit(x_train, y_train)
    valid = pd.read_csv(valid_path)
    valid['intercept'] = 1.0
    x_eval, y_val = valid[['intercept', 'x_1', 'x_2']], valid[['y']].values.ravel()
    predictions = clf.predict(x_eval)
    np.savetxt(save_path, predictions)
    print(clf.coef_[0])
    util.plot(x_eval.values, y_val, clf.coef_[0], save_img)


if __name__ == '__main__':
    main(train_path='ds1_train.csv', valid_path='ds1_valid.csv', save_path='vscikit-learn_reg_pred_1.txt', save_img='vscikit-learn_logreg_pred_1.png')
    main(train_path='ds2_train.csv', valid_path='ds2_valid.csv', save_path='vscikit-learn_logreg_pred_2.txt', save_img='vscikit-learn_logreg_pred_2.png')
