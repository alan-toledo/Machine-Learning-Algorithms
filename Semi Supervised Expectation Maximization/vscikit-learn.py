import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import mixture

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)

def main(is_semi_supervised, trial_num):
    """EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'.format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    g = mixture.GaussianMixture(n_components=K)
    g.fit(x) 
    plot_gmm_preds(x, g.predict(x), is_semi_supervised, plot_id=trial_num)


# Helper functions
# *** START CODE HERE ***
def compute_likelihood(n, x, w, mu, sigma, phi):
    ll = 0
    for k in range(K):
        for i in range(n):
            likelihood = np.exp(-0.5*(np.dot(np.dot((x[i] - mu[k]).T, np.linalg.pinv(sigma[k])), (x[i] - mu[k]))))
            likelihood = np.true_divide(likelihood, np.power(2.0*np.pi, 0.5*x.shape[1])*np.power(np.linalg.det(sigma[k]), 0.5))
            w[i,k] = phi[k]*likelihood
            ll = ll + np.log(w[i,k].clip(min=1e-10))
    w /= np.sum(w, axis=1, keepdims=True)
    return w, ll
# *** END CODE HERE ***

def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'vscikit-learn_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)

def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z

if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    #NUM_TRIALS
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)
