'''
Sebastian Ballesteros
'''

'''
'''

############################## IMPORTS  ########################################
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import sys
import nltk
import numpy as np
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk import word_tokenize

from matplotlib import pyplot as plt
import math
import pandas
import codecs
import string

########################### GLOBAL CONSTANTS ##################################
LEMMATIZE = False
SMOOTHING = 1
STEM = False
MIN_FREQUENCY = 2
TRAIN_PERCENTAGE = 0.9
TEST_PERCENTAGE = 0.1
SENTENCES_COUNT = 10662
DATA_DIRECTORY = r'./rt-polaritydata'

###########################  HELPER FUNCTIONS ##################################
def read_file(filename):
    with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as file:
             return file.readlines()

def tokenize(sentence):
    sentence_without_punctuation = (''.join([word for word in sentence if word not in string.punctuation]))
    sentence_tokenized = nltk.word_tokenize(sentence_without_punctuation)
    return sentence_tokenized

def stem(sentences):
    stemmer = PorterStemmer()

    sentences_stemmed = []
    for sentence in sentences:
        sentences_stemmed.append(''.join([stemmer.stem(word) for word in sentence]))
    return sentences_stemmed

def lemmatize(sentences):
    lemmatizer = WordNetLemmatizer()

    sentences_lemmatized = []
    for sentence in sentences:
        sentences_lemmatized.append(''.join([lemmatizer.lemmatize(word) for word in sentence]))
    return sentences_lemmatized

def remove_stop_words(tokenized_sentencees):
    stop_words = stopwords.words('english')

    new_tokenized_sentences = []
    for sentence in tokenized_sentencees:
        without_stop_words = [word for word in sentence if word not in stop_words]
        new_tokenized_sentences.append(without_stop_words)
    return new_tokenized_sentences


###########################  MAIN FUNCTIONS ##################################

def load_data():
    negative_sentences = read_file('./rt-polaritydata/rt-polarity.neg')
    positive_sentences = read_file('./rt-polaritydata/rt-polarity.pos')

    y_negative = np.zeros((len(negative_sentences), 1))
    y_positive = np.ones((len(positive_sentences), 1))

    y = np.concatenate((y_negative, y_positive))
    x = np.concatenate((negative_sentences, positive_sentences))

    return(x, y)


def preprocess(x, y):

    #lemmatize or stem
    if(LEMMATIZE):
        x = lemmatize(x)
    elif(STEM):
        x = stem(x)

    #Separate dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=TEST_PERCENTAGE)

    #extract features and remove infrequently occurring words
    tokenizer = CountVectorizer(min_df=MIN_FREQUENCY, tokenizer=nltk.word_tokenize, stop_words='english')
    x_counts = tokenizer.fit_transform(x_train)

    # Convert raw frequency counts into TF-IDF (Term Frequency -- Inverse Document Frequency) values
    fooTfmer = TfidfTransformer()
    x_train_vec = fooTfmer.fit_transform(x_counts)

    #print(x)

    x_test_vec = tokenizer.transform(x_test)

    return x_train_vec, x_test_vec, y_train, y_test


def train_naive_bayes(x_train, y_train, x_test):
    naive_bayes = MultinomialNB(alpha=SMOOTHING)

    naive_bayes.fit(x_train, y_train.ravel())

    naive_bayes.score(x_train, y_train)

    y_pred = naive_bayes.predict(x_test)

    return y_pred

def test_naive_bayes(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred) * 100
    print('-----------Naive Bayes-------------')
    print("Accuracy: {:.2f}%".format(accuracy))
    print("Confusion Matrix: \n {}".format(confusion_matrix(y_test, y_pred)))
    return accuracy

def train_logistic_regression(x_train, y_train, x_test):
    logistic_regression = LogisticRegression()

    logistic_regression.fit(x_train, y_train.ravel())

    logistic_regression.score(x_train, y_train)

    y_pred = logistic_regression.predict(x_test)

    return y_pred

def test_logistic_regression(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred) * 100
    print('-------Logistic Regression--------')
    print("Accuracy: {:.2f}%".format(accuracy))
    print("Confusion Matrix: \n {}".format(confusion_matrix(y_test, y_pred)))
    return accuracy

def train_support_vector(x_train, y_train, x_test):
    support_vector = LinearSVC()

    support_vector.fit(x_train, y_train.ravel())

    support_vector.score(x_train, y_train)

    y_pred = support_vector.predict(x_test)

    return y_pred

def test_support_vector(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred) * 100
    print('-----Support Vector Machines------')
    print("Accuracy: {:.2f}%".format(accuracy))
    print("Confusion Matrix: \n {}".format(confusion_matrix(y_test, y_pred)))
    return accuracy

def train_bernoulli(x_train, y_train, x_test):
    bernoulli = BernoulliNB()

    bernoulli.fit(x_train, y_train.ravel())

    bernoulli.score(x_train, y_train)

    y_pred = bernoulli.predict(x_test)

    return y_pred

def test_bernoulli(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred) * 100
    print('-------------Bernoulli------------')
    print("Accuracy: {:.2f}%".format(accuracy))
    print("Confusion Matrix: \n {}".format(confusion_matrix(y_test, y_pred)))
    return accuracy

def train_random():
    y_pred = np.random.randint(2, size=(math.ceil(SENTENCES_COUNT * TEST_PERCENTAGE)))
    return y_pred

def test_random(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred) * 100
    print('---------------Random-------------')
    print("Accuracy: {:.2f}%".format(accuracy))
    print("Confusion Matrix: \n {}".format(confusion_matrix(y_test, y_pred)))
    return accuracy

def plot_results(models_accuracy):
    fig, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(16,9))

    x1 = np.arange(len(models_accuracy))

    ax1.bar(x1, [x[1] for x in models_accuracy])

    # Place values on top of bars
    for i, v in enumerate(models_accuracy):
        #(x, y, text)
        ax1.text(i - 0.25, 40, v[0])
        ax1.text(i - 0.25, v[1] + 2, str(v[1]) + '%')

    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Models')
    ax1.set_ylim([0, 100])

    plt.show()

def main():
    x, y = load_data()

    x_train, x_test, y_train, y_test = preprocess(x, y)

    models_accuracy = []

    y_pred = train_naive_bayes(x_train, y_train, x_test)
    accuracy =test_naive_bayes(y_test, y_pred)
    models_accuracy.append(('Naive Bayes', accuracy))

    y_pred = train_logistic_regression(x_train, y_train, x_test)
    accuracy = test_logistic_regression(y_test, y_pred)
    models_accuracy.append(('Logistic Regression', accuracy))

    y_pred = train_support_vector(x_train, y_train, x_test)
    accuracy = test_support_vector(y_test, y_pred)
    models_accuracy.append(('Support Vectors', accuracy))

    y_pred = train_bernoulli(x_train, y_train, x_test)
    accuracy = test_bernoulli(y_test, y_pred)
    models_accuracy.append(('Bernoulli', accuracy))

    y_pred = train_random()
    accuracy = test_random(y_test, y_pred)
    models_accuracy.append(('Random', accuracy))

    plot_results(models_accuracy)

if __name__ == "__main__":
    main()
