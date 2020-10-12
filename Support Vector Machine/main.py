import collections
import numpy as np
import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For simplicity, you should split on whitespace, not
    punctuation or any other character. For normalization, you should convert
    everything to lowercase.  Please do not consider the empty string (" ") to be a word.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """
    words = [str(word).lower() for word in message.split(' ') if word.strip() != ""]
    return words

def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    dictionary = {}
    full_words = []
    for message in messages:
        words = get_words(message)
        full_words = full_words + list(set(words))
    index = 0
    dict_counter = collections.Counter(full_words)
    dict_counter_ordered = collections.OrderedDict(dict_counter.most_common())
    for word in dict_counter_ordered:
        if dict_counter_ordered[word] >= 5:
            dictionary[word] = index
            index = index + 1
    return dictionary

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    matrix = np.zeros((len(messages), len(word_dictionary)))
    for row_i, message in enumerate(messages):
        words = get_words(message)
        
        dictionary = {}

        for word in words:
            if word not in dictionary:
                dictionary[word] = 1
            else:
                dictionary[word] = dictionary[word] + 1

        for word in words:
            if word in word_dictionary:
                col_i = word_dictionary[word]
                matrix[row_i][col_i] = dictionary[word]
    
    return matrix

def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    SVM_radio = (-1.0, -1.0)
    for radio in radius_to_consider:
        svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radio)
        svm_accuracy = np.mean(svm_predictions == val_labels)
        if svm_accuracy > SVM_radio[1]:
            SVM_radio = (radio, svm_accuracy)
    return SVM_radio[0]


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))
    
    train_matrix = transform_text(train_messages, dictionary)
    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)
    np.savetxt('svm_predictions.txt', svm_predictions)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy))


if __name__ == "__main__":
    main()
