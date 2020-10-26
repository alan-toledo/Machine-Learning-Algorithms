import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pdb
import pandas as pd

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

def df_to_dataset(dataframe, labels, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

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
    

    model = keras.Sequential([
                keras.Input(shape=(784,)),
                keras.layers.Dense(300, activation='sigmoid'),
                keras.layers.Dense(10, activation='softmax')
            ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    n_batch = 1000
 
    training_history = model.fit(train_data, train_labels, validation_data=(dev_data, dev_labels), batch_size=n_batch, epochs=args.num_epochs)
    training_results = pd.DataFrame(training_history.history)
    training_results['epoch'] = training_history.epoch


    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(training_results['epoch'], training_results['loss'],'r', label='train')
    ax1.plot(training_results['epoch'], training_results['val_loss'], 'b', label='dev')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.set_title('Without Regularization')
    ax1.legend()
    ax2.plot(training_results['epoch'], training_results['accuracy'], 'r', label='train')
    ax2.plot(training_results['epoch'], training_results['val_accuracy'], 'b', label='dev')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('accuracy')
    ax2.legend()

    fig.savefig('./' + 'vTensorFlow' + '.pdf')

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    
if __name__ == '__main__':
    main()
