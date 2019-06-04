from typing import Dict, Any

import pandas as pd
import numpy as np
import math
import operator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)


def knn(trainingSet, testInstance, k):
    distances = {}
    sort = {}
    length = testInstance.shape[1]
    # print("x= ", trainingSet.iloc[2])

    for x in range(len(trainingSet)):

        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)
        # print('dist ', dist)
        # print('dist[0]=', dist[0])
        distances[x] = dist[0]

        # print(distances[x])

    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))

    neighbors = []

    for x in range(k):
        neighbors.append(sorted_d[x][0])

    classCnt = {}

    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]

        if response in classCnt:
            classCnt[response] += 1
        else:
            classCnt[response] = 1

    sorted_cnt = sorted(classCnt.items(), key=operator.itemgetter(1), reverse=True)
    return (sorted_cnt[0][0], neighbors)

# data = pd.read_csv("train_dataset.csv")
# dftest = data[['reviews.rating','positive_words','negative_words','reviews.doRecommend']]
data = pd.read_csv("train_dataset.csv")
df = data[['reviews.rating','polarity','reviews.doRecommend']]


testdata = pd.read_csv("test_dataset.csv")

dflabel= testdata['reviews.doRecommend']
#.astype(int)
dflabel_list=dflabel.values.tolist()

dftestdata = testdata[['reviews.rating','polarity']]
testdata_list=dftestdata.values.tolist()

k = 8
predicted_data=[]

for x in testdata_list:
    print('x=  ',x)
    testSet=x
    test = pd.DataFrame(testSet)
    # print('testInstance = ', test)
    result, neigh = knn(df, test, k)
    predicted_data.append(result)
    # print(result)
# print(predicted_data)

dftestdata['KNN_predicted_labels']=pd.DataFrame(predicted_data)
dftestdata.to_csv("KNN_dataset.csv",index=False)

results = confusion_matrix(dflabel_list, predicted_data)
print('Confusion Matrix:')
print(results)
print('Accuracy Score :',accuracy_score(dflabel_list, predicted_data))
