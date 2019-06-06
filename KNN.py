from typing import Dict, Any

import pandas as pd
import numpy as np
import math
import operator
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)


def knn(fit_set, predict_instance, k):
    distances = {}
    length = predict_instance.shape[1]
    for x in range(len(fit_set)):
        dist = euclideanDistance(predict_instance, fit_set.iloc[x], length)
        distances[x] = dist[0]
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))

    neighbors = []

    for x in range(k):
        neighbors.append(sorted_d[x][0])

    classCnt = {}

    for x in range(len(neighbors)):
        response = fit_set.iloc[neighbors[x]][-1]

        if response in classCnt:
            classCnt[response] += 1
        else:
            classCnt[response] = 1

    sorted_cnt = sorted(classCnt.items(), key=operator.itemgetter(1), reverse=True)
    return (sorted_cnt[0][0], neighbors)
data = pd.read_csv("train_dataset.csv")
fit = data[['reviews.rating','polarity','reviews.doRecommend']]


testdata = pd.read_csv("test_dataset.csv")

df= testdata['reviews.doRecommend']
#.astype(int)
df_list=df.values.tolist()

dftestdata = testdata[['reviews.rating','polarity']]
testdata_list=dftestdata.values.tolist()


k = 8
predicted_data=[]

for x in testdata_list:
    print('x=  ',x)
    testSet=x
    predict = pd.DataFrame(testSet)
    result, neigh = knn(fit, predict, k)
    predicted_data.append(result)

dftestdata['reviews.doRecommend']=testdata['reviews.doRecommend']
dftestdata['KNN_predicted_class']=pd.DataFrame(predicted_data)
dftestdata.to_csv("test_dataset.csv",index=False)

results = confusion_matrix(df_list, predicted_data)
print('Confusion Matrix:')
print(results)
print('Accuracy Score :',accuracy_score(df_list, predicted_data))
print('Classification Report')
print(classification_report(df_list, predicted_data))
