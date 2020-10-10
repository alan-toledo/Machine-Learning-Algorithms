import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import util

def main(train_path, valid_path, save_path, save_img):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
        save_img: Path to save plot linear decision boundary
    """
    train = pd.read_csv(train_path)
    x_train, y_train = train[['x_1', 'x_2']], train[['y']].values.ravel()
    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train, y_train)
    valid = pd.read_csv(valid_path)
    x_eval, y_val = valid[['x_1', 'x_2']], valid[['y']].values.ravel()
    predictions = clf.predict(x_eval)
    np.savetxt(save_path, predictions)
    print(clf.intercept_[0], clf.coef_[0])
    util.plot(x_eval.values, y_val, clf.intercept_[0], clf.coef_[0], save_img)
   

if __name__ == '__main__':
    main(train_path='ds1_train.csv', valid_path='ds1_valid.csv', save_path='vscikit-learn_gda_pred_1.txt',  save_img='vscikit-learn_gda_pred_1.png')
    main(train_path='ds2_train.csv', valid_path='ds2_valid.csv', save_path='vscikit-learn_gda_pred_2.txt', save_img='vscikit-learn_gda_pred_2.png')
