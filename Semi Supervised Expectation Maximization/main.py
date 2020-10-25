import matplotlib.pyplot as plt
import numpy as np
import os
import pdb

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
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    n = x.shape[0]
    indices = np.random.randint(K, size=n)
    mu, sigma = [],[]
    for k in range(K):
        x_filtered = x[indices == k]
        mean = np.mean(x_filtered, axis=0)
        cova = np.cov(x_filtered.T)
        mu.append(mean)
        sigma.append(cova)
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full((K), float(1/K))
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (n, K)
    w = np.full((n, K), float(1/K))
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)

def run_em(x, w, phi, mu, sigma, max_iter=1000):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim)
        max_iter: Max iterations. No need to change this

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    n = int(x.shape[0])
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code

        # (1) E-step: Update your estimates in w

        # (2) M-step: Update the model parameters phi, mu, and sigma

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

        # *** START CODE HERE ***
        print('EM Algorithm (unsupervised)', it)

        if ll is not None:                
            prev_ll = ll

        w, ll = compute_likelihood(n, x, w, mu, sigma, phi)
     
        for k in range(K):
            sum_k = w[:,k].sum()   
            phi[k] = np.true_divide(sum_k, float(n))
            mu[k] = np.true_divide(np.dot(w[:,k], x), sum_k)
            sigma[k] = 0.0
            for i in range(n):
                x_mu = np.reshape(x[i] - mu[k], (mu[k].shape[0], 1))
                sigma[k] =  sigma[k] + w[i,k]*np.dot(x_mu, x_mu.T)/sum_k
        it = it + 1
        # *** END CODE HERE ***
    return w

def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma, max_iter=1000):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim)
        max_iter: Max iterations. No need to change this

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    n = int(x.shape[0])
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code

        # (1) E-step: Update your estimates in w

        # (2) M-step: Update the model parameters phi, mu, and sigma

        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

        # *** START CODE HERE ***
        if ll is not None:                
            prev_ll = ll

        print('Semi-Supervised EM Algorithm', it)
        w, ll = compute_likelihood(n, x, w, mu, sigma, phi)
        
        for k in range(K):
            x_tilde_filtered = [y for x, y in zip(z_tilde == k, x_tilde) if x == True]
            x_tilde_filtered = np.array(x_tilde_filtered)
            for i in range(x_tilde_filtered.shape[0]):
                likelihood = np.exp(-0.5*(np.dot(np.dot((x_tilde_filtered[i] - mu[k]).T, np.linalg.pinv(sigma[k])), x_tilde_filtered[i] - mu[k])))
                likelihood = np.true_divide(likelihood, np.power(2.0*np.pi, x_tilde_filtered.shape[1]/2)*np.power(np.linalg.det(sigma[k]), 0.5)).flatten()
                ll = ll + alpha*np.log(phi[k]*likelihood.clip(min=1e-10))
        
        n_tilde = z_tilde.shape[0]
        for k in range(K):
            sum_tilde = alpha*np.sum(z_tilde == k)
            sum_wk = w[:,k].sum()
            x_tilde_filtered = np.array([y for x, y in zip(z_tilde == k, x_tilde) if x == True])
            phi[k] = np.true_divide(sum_wk + sum_tilde, n + alpha*n_tilde)        
            mu[k] = np.true_divide(np.dot(w[:,k], x) + alpha*x_tilde_filtered.sum(axis=0), sum_wk +  sum_tilde)

            temp = 0.0
            for i in range(n):
                x_mu = np.reshape(x[i] - mu[k], (mu[k].shape[0], 1))
                temp = temp + w[i,k]*np.dot(x_mu, x_mu.T)
            
            for i in range(x_tilde_filtered.shape[0]):
                x_mu = np.reshape(x_tilde_filtered[i] - mu[k], (mu[k].shape[0], 1))
                temp = temp + alpha*np.dot(x_mu, x_mu.T)
            
            sigma[k] = np.true_divide(temp, sum_wk + sum_tilde)
        it = it + 1
        # *** END CODE HERE ***
    return w

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

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
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
        # Once you've implemented the semi-supervised version,
        # uncomment the following line:
        main(is_semi_supervised=True, trial_num=t)
