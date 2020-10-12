import collections
import numpy as np
import util

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

    # *** START CODE HERE ***
    words = [str(word).lower() for word in message.split(' ') if word.strip() != ""]
    return words
    # *** END CODE HERE ***

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

def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    p_spam = np.sum(labels == 1.0)/len(labels)
    p_ham =  np.sum(labels == 0.0)/len(labels)
    V = matrix.shape[1]
    spam, ham = {}, {}
    words_spam = np.sum(matrix[labels == 1])
    words_ham = np.sum(matrix[labels == 0])
    n_words_by_columns = matrix.sum(axis=0)
    for col in range(V):
        word_given_spam = np.sum(matrix[labels == 1][:,col])        
        word_given_ham = n_words_by_columns[col] - word_given_spam
        
        spam[col] = float(word_given_spam + 1.0) / float(words_spam + V)
        ham[col] = float(word_given_ham + 1.0) / float(words_ham + V)

    return p_spam, p_ham, spam, ham

def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    p_spam, p_ham, spam, ham = model
    predictions = []
    for words in matrix:

        prediction = 0.0
        is_spam = p_spam
        is_ham = p_ham

        for col in range(matrix.shape[1]):
            is_spam *= np.power(spam[col], words[col])
            is_ham *= np.power(ham[col], words[col])
       
        if is_spam > is_ham:
            prediction = 1.0

        predictions.append(prediction)
    return np.array(predictions)

def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    _, _, spam, ham = model
    most_indicative_words = {}
    for word in dictionary:
        col = dictionary[word]
        most_indicative_words[word] = np.log(float(spam[col])/float(ham[col]))
    
    sorted_most_indicative_words = {k: v for k, v in sorted(most_indicative_words.items(), reverse=True, key=lambda item: item[1])}
    top_five = []
    for word in sorted_most_indicative_words:
        top_five.append(word)
        if len(top_five) >= 5:
            break
    return list(top_five)

def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))
    
    train_matrix = transform_text(train_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)
    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)
    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)
    
    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)
    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))
    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

if __name__ == "__main__":
    main()
