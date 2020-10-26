import argparse
import functools
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    """
    Compute softmax function for a batch of input values. 
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***
    tempX = x - (np.max(x, axis=1)).reshape(x.shape[0], 1)
    p = np.exp(tempX) / np.sum(np.exp(tempX), axis=1, keepdims=True)
    return p
    # *** END CODE HERE ***

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***
    return 1/(1 + np.exp(-x))
    # *** END CODE HERE ***

def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***
    dict_mapping = {}
    dict_mapping['W1'] = np.random.normal(loc=0.0, scale=1.0, size=(input_size, num_hidden))
    dict_mapping['b1'] = np.zeros(num_hidden)
    dict_mapping['W2'] = np.random.normal(loc=0.0, scale=1.0, size=(num_hidden, num_output))
    dict_mapping['b2'] = np.zeros(num_output)
    return dict_mapping
    # *** END CODE HERE ***

def forward_prop(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    #data (1.000 x 784) #W1 (784 x 300) #b1 (300,1) 
    xW1_b1 = (data).dot(params['W1']) + params['b1'] #(1.000, 300)
    z = sigmoid(xW1_b1) # (1.000, 300)
    
    #W2 (300 x 10) #b2 (10,1)
    zW2_b2 = (z).dot(params['W2']) +  params['b2'] #(1.000, 10)
    output = softmax(zW2_b2) #(1.000, 10)
    #labels (1.000, 10)
    average_loss = 0.0
    n = labels.shape[0]
    K = labels.shape[1]
    for i in range(n):
        for j in range(K):
            average_loss = average_loss + labels[i,j]*np.log(output[i, j])
    average_loss = -float(average_loss)/float(n)
    return (z, output, average_loss)
    # *** END CODE HERE ***

def backward_prop(data, labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    (z, output, _) = forward_prop(data, labels, params)
    gradients = {}

    dscores = output - labels
    dscores = np.true_divide(dscores, float(labels.shape[0])) #(1000, 10)

    #(300, 1000) * (1000, 10)
    gradients['W2'] = np.dot(z.T, dscores) #(300, 10)
    gradients['b2'] = np.sum(dscores, axis=0, keepdims=True) #(10,1)

    #(1.000, 10) * (10, 300)
    dhidden = np.dot(dscores, params['W2'].T) #(1000, 300)
    dhidden = np.multiply(dhidden, z*(1.0 - z)) #(1000, 300) * (1000, 300)
    #(784, 1000) * (1000, 300)
    gradients['W1'] = np.dot(data.T, dhidden) #(784, 300)
    gradients['b1'] = np.sum(dhidden, axis=0, keepdims=True) #(300, 1)
    return gradients
    # *** END CODE HERE ***

def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    (z, output, _) = forward_prop(data, labels, params)
    gradients = {}

    dscores = output - labels
    dscores = np.true_divide(dscores, float(labels.shape[0]))
    dCE = dscores
    #(300, 1000) * (1000, 10)
    gradients['W2'] = np.dot(z.T, dCE) #(300, 10)
    gradients['W2'] = gradients['W2'] + 2.0*float(reg)*params['W2'] #(300, 10)
    gradients['b2'] = np.sum(dCE, axis=0, keepdims=True) #(10,1)

    #(1.000, 10) * (10, 300)
    dhidden = np.dot(dCE, params['W2'].T) #(1.000, 300)
    dhidden = np.multiply(dhidden, z*(1.0 - z)) #(1000, 300) * (1000, 300)
    #(784, 1000) * (1000, 300)
    gradients['W1'] = np.dot(data.T, dhidden) #(784, 300)
    gradients['W1'] = gradients['W1'] +  2.0*float(reg)*params['W1']
    gradients['b1'] = np.sum(dhidden, axis=0, keepdims=True) #(300, 1)
    return gradients
    # *** END CODE HERE ***

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    # *** START CODE HERE ***
    # back prop to calculate the gradients
    for i in range(50):
        gradients = backward_prop_func(train_data[batch_size*i:batch_size*(i+1)], train_labels[batch_size*i:batch_size*(i+1)], params, forward_prop_func)
        params['W1'] = params['W1'] - float(learning_rate)*gradients['W1'].reshape(params['W1'].shape)
        params['b1'] = params['b1'] - float(learning_rate)*gradients['b1'].reshape(params['b1'].shape)
        params['W2'] = params['W2'] - float(learning_rate)*gradients['W2'].reshape(params['W2'].shape)
        params['b2'] = params['b2'] - float(learning_rate)*gradients['b2'].reshape(params['b2'].shape)
    # *** END CODE HERE ***
    # This function does not return anything
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):
    """
    Train model using gradient descent for specified number of epochs.
    
    Evaluates cost and accuracy on training and dev set at the end of each epoch.

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        dev_data: A numpy array containing the dev data
        dev_labels: A numpy array containing the dev labels
        get_initial_params_func: A function to initialize model parameters
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API
        num_hidden: Number of hidden layers
        learning_rate: The learning rate
        num_epochs: Number of epochs to train for
        batch_size: The amount of items to process in each batch

    Returns: 
        params: A dict of parameter names to parameter values for the trained model
        cost_train: An array of training costs at the end of each training epoch
        cost_dev: An array of dev set costs at the end of each training epoch
        accuracy_train: An array of training accuracies at the end of each training epoch
        accuracy_dev: An array of dev set accuracies at the end of each training epoch
    """

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func)
        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        print(epoch, cost, compute_accuracy(output, train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    """Predict labels and compute accuracy for held-out test data"""
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == 
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    """Convert labels from integers to one hot encoding"""
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file):
    """Load images and labels"""
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    """Trains model, applies model to test data, and (optionally) plots loss"""
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )
    
    t = np.arange(num_epochs)
    
    np.save("{}.npy".format(name), params)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))
    
    return accuracy

def main(plot=True):
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    np.random.seed(100)
    train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(60000)
    train_data = train_data[p,:]
    train_labels = train_labels[p,:]
    dev_data = train_data[0:10000,:]
    dev_labels = train_labels[0:10000,:]
    train_data = train_data[10000:,:]
    train_labels = train_labels[10000:,:]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std
    
    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }

    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels, functools.partial(backward_prop_regularized, reg=0.0001), args.num_epochs, plot)

    return baseline_acc, reg_acc

    
if __name__ == '__main__':
    main()
