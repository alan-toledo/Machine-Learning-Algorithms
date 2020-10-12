import collections
import numpy as np
import util
from sklearn.naive_bayes import GaussianNB

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

def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))
    
    train_matrix = transform_text(train_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)
    gnb = GaussianNB()
    naive_bayes_predictions = gnb.fit(train_matrix, train_labels).predict(test_matrix)
    np.savetxt('vscikit-learn_spam_naive_bayes_predictions', naive_bayes_predictions)
    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)
    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

if __name__ == "__main__":
    main()
