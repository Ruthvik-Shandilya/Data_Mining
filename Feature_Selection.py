import codecs

import pandas as pd
import numpy as np
import nltk
from nltk import collections
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from svm import SVM


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


lem = WordNetLemmatizer()
#stemmer= PorterStemmer()

df = pd.read_csv(r"Dataset_preprocessed.csv")


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t,"v") for t in word_tokenize(articles)]

cv = CountVectorizer(tokenizer=LemmaTokenizer(),strip_accents = 'unicode', lowercase = True )

text_counts = cv.fit_transform(df['review_without_stopwords'])
# word_freq = dict(zip(cv.get_feature_names(),np.asarray(text_counts.sum(axis=0)).ravel()))
# word_counter = collections.Counter(word_freq)
# print(word_counter)
# word_count_df = pd.DataFrame(word_counter.most_common(20), columns=["word","freq"])
#
#

X = df[['polarity','reviews.rating']]
y = df['reviews.doRecommend']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# export_csv = word_count_df.to_csv (r'CountVectorizer.csv', index = None, header=True)
# h = codecs.open("CountVectorizer.csv","r","utf-8")
# samplewords = h.read()
# h.close()
#
#
# vocab = np.array(text_counts)
#
# print(vocab)


# df1=df[['reviews.doRecommend','positive_words','negative_words']]
#
# train,test = train_test_split(
# df1, test_size=0.2, random_state=1)

# df2 = pd.DataFrame(train)
# df2.to_csv("train_dataset.csv",index=False)
#
# df3 = pd.DataFrame(test)
# df3.to_csv("test_dataset.csv",index=False)


# i=1
# b=np.array(train.iloc[:,i:i+2])
# c=np.array(test.iloc[:,i:i+2])
# print(Y_test)
# c=np.array(test.iloc[:,i:i+2])
# print("printing array value : " ,b)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train,y_train)
y_pred = svclassifier.predict(X_test)

# svclassifier=SVM()
# svclassifier.fit(b)
# y_pred=svclassifier.predict(c)
# #
print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

print('Accuracy Score :',accuracy_score(y_test, y_pred))


# def stemming_tokenizer(str_input):
#     words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
#     words = [stemmer.stem(word) for word in words]
#     return words




# cv = CountVectorizer(tokenizer=stemming_tokenizer)
#
# X = cv.fit_transform(df['review_without_stopwords'])
#
# df4 = pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
# print(df4)
#
# export_csv = df4.to_csv (r'stemming.csv', index = None, header=True)
# s = codecs.open("stemming.csv","r","utf-8")
# samplewords = s.read()
# s.close()


# vec = TfidfVectorizer(stop_words='english',
#                       tokenizer=LemmaTokenizer(),
#                       use_idf=False,
#                       norm='l1')
#
# matrix = vec.fit_transform(df['reviews'])
# df5 = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names())
# print(df5)
#
# export_csv = df5.to_csv (r'TF_IDF.csv', index = None, header=True)
# v = codecs.open("TF_IDF.csv","r","utf-8")
# samplewords = v.read()
# v.close()


# print(X)
# print(cv.get_feature_names())
# print(X.toarray())