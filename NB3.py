import os
import string
from typing import TextIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# In[2]:


X = []  # an element of X is preprocessed reviews
Y = []  # an element of Y represents the  labels of the corresponding X element


with open('Dataset_preprocessed.csv', "r",encoding="utf8") as f:
            reader = csv.reader(f)
            included_cols = [5,13]   # here 2 column numbers for reviews and labels from preproceesed dataset csv


            for row in reader:
                content = list(row[i] for i in included_cols)
                # print(content[0])
                # print(content1)
                X.append(content[1])
                Y.append(content[0])
                # print(X)
                # print(Y)


# In[3]:


train = pd.read_csv(r"train_dataset.csv")
X_train = train['review_without_stopwords']
Y_train = train['reviews.doRecommend']

test = pd.read_csv(r"train_dataset.csv")
X_test = train['review_without_stopwords']
Y_test = train['reviews.doRecommend']

# print("xtrain is:",X_train)


# In[5]:

# Building a vocabulary of words from the given documents
vocab = {}
for i in range(len(X_train)):
    word_list = []
    for word in X_train[i].split():
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1

            # In[6]:
# print(vocab)


features = []
for key in vocab:
    features.append(key)


# for key in vocab:
#         if vocab[key] >= cutoff_freq:
#             features.append(key)

# print("features are:::::::",features)


# In[9]:


# To represent training data as word vector counts
X_train_dataset = np.zeros((len(X_train), len(features)))
# This can take some time to complete
for i in range(len(X_train)):
    # print(i) # Uncomment to see progress
    word_list = [word.strip(string.punctuation).lower() for word in X_train[i][1].split()]
    for word in word_list:
        if word in features:
            X_train_dataset[i][features.index(word)] += 1

# In[10]:


# To represent test data as word vector counts
X_test_dataset = np.zeros((len(X_test), len(features)))
# This can take some time to complete
for i in range(len(X_test)):
    # print(i) # Uncomment to see progress
    word_list = [word.strip(string.punctuation).lower() for word in X_test[i][1].split()]
    for word in word_list:
        if word in features:
            X_test_dataset[i][features.index(word)] += 1


# In[12]:


# Implementing Multinomial Naive Bayes from scratch
class MultinomialNaiveBayes:

    def __init__(self):

        self.count = {}
        # classes represents the labels
        self.classes = None

    def fit(self, X_train, Y_train):
        # This can take some time to complete
        self.classes = set(Y_train)
        for class_ in self.classes:
            self.count[class_] = {}
            for i in range(len(X_train[0])):
                self.count[class_][i] = 0
            self.count[class_]['total'] = 0
        self.count['total_points'] = len(X_train)

        for i in range(len(X_train)):
            for j in range(len(X_train[0])):
                self.count[Y_train[i]][j] += X_train[i][j]
            self.count[Y_train[i]]['total'] += 1

    def __probability(self, test_point, class_):

        log_prob = np.log(self.count[class_]['total']) - np.log(self.count['total_points'])
        total_words = len(test_point)
        for i in range(len(test_point)):
            current_word_prob = test_point[i] * (
                        np.log(self.count[class_][i] + 1) - np.log(self.count[class_]['total'] + total_words))
            log_prob += current_word_prob

        return log_prob

    def __predictSinglePoint(self, test_point):

        best_class = None
        best_prob = None
        first_run = True

        for class_ in self.classes:
            log_probability_current_class = self.__probability(test_point, class_)
            if (first_run) or (log_probability_current_class > best_prob):
                best_class = class_
                best_prob = log_probability_current_class
                first_run = False

        return best_class

    def predict(self, X_test):
        # This can take some time to complete
        Y_pred = []
        for i in range(len(X_test)):
            # print(i) # Uncomment to see progress
            Y_pred.append(self.__predictSinglePoint(X_test[i]))

        return Y_pred

    def score(self, Y_pred, Y_true):
        # returns the mean accuracy
        count = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y_true[i]:
                count += 1
        return count / len(Y_pred)


# In[13]:


mnb = MultinomialNaiveBayes()
mnb.fit(X_train_dataset, Y_train)

Y_test_pred = mnb.predict(X_test_dataset)

our_score_test = mnb.score(Y_test_pred, Y_test)

print("Our score on testing data :", our_score_test)
print("Classification report for testing data :-")



# Using sklearn's Multinomial Naive Bayes
NB = MultinomialNB()
NB.fit(X_train_dataset, Y_train)
Y_test_pred = NB.predict(X_test_dataset)

sklearn_score_train = NB.score(X_train_dataset, Y_train)
print("Sklearn's score on training data :", sklearn_score_train)

sklearn_score_test = NB.score(X_test_dataset, Y_test)
print("Sklearn's score on testing data :", sklearn_score_test)

print("Classification report for testing data :-")

# In[14]:

# print("Score of our model on test data:", our_score_test)
# print("Score of inbuilt sklearn's MultinomialNB on the same data :", sklearn_score_test)

print(classification_report(Y_test, Y_test_pred))