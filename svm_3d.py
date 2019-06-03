import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
from itertools import islice
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1, 1 , 1],
                              [1, 1, -1],
                              [-1, 1, -1],
                              [-1, -1, -1],
                              [1, -1, -1],
                              [1, -1, 1],
                              [-1, 1, 1],
                              [-1, -1, 1]]


        all_data = []
        for yi in self.data:
            #print(yi)
            #print(self.data)
            for featureset in self.data[yi]:
                #print(featureset)
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        #print(self.max_feature_value)
        #print(self.min_feature_value)
        # support vectors yi(xi.w+b) = 1


        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      ]

        # extremely expensive
        b_range_multiple = 0.4
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 3
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum,latest_optimum])
            #print('w:',w)
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):

                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        for i in self.data:

                            for xi in self.data[i]:
                                yi = i

                                if yi * (np.dot(w_t, xi) + b) < 1:
                                    found_option = False
                                break

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                        break

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            # ||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2


    def predict(self, features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        # if classification != 0 and self.visualization:
        #     self.ax.scatter(features[0], features[1],features[2], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


data_dict = {-1: np.array([[-0.105555555555555,1],[0,2],[0.1093,3],[0.2333,3],[-0.333,3],[0.185,3],[0.5,3],[-0.391,3],[0,3],[0,3],[0,4],[0.1,3],[-0.6,1],[-0.6,1]
                            ]),

             1: np.array([[0.185,5],[0.5,4],[0.8,5],[0,5],[0.3291,5],[0.2,5],[0.5444,5],[0.3666,5],[0.3345,4],[0.6,4]
                           ])}


header = ["reviews.doRecommend","positive_words","negative_words","reviews.rating"]
data_read = pd.read_csv(r"train_dataset.csv",sep=',', usecols=header)

data_read1=data_read[["reviews.doRecommend","positive_words","negative_words","reviews.rating"]]

my_dict={}
list=[]
list2=[]
for index,row in data_read1.iterrows():
    if row['reviews.doRecommend']==1:
        list.append(row[1:].tolist())
    else:
        list2.append(row[1:].tolist())

my_dict[1]=np.asarray(list)
my_dict[-1]=np.asarray(list2)

svm = Support_Vector_Machine()
svm.fit(data=my_dict)

header = ["reviews.doRecommend","positive_words","negative_words","reviews.rating"]
data_read2 = pd.read_csv(r"test_dataset.csv",sep=',', usecols=header)

data_read3=data_read2[["reviews.doRecommend","positive_words","negative_words","reviews.rating"]]
list3 = []
list4 = []
for index,row in data_read3.iterrows():
        # print(row[0:].tolist())
        list3.append(row[1:].tolist())

for index,row in data_read3.iterrows():
        # print(row[0:].tolist())
        list4.append(row[0:1].tolist())

predict_us = list3
pred = []

for p in predict_us:
   pred.append(svm.predict(p))

print('pred:',pred)
print('list3:',list3)

#svm.visualize()

results = confusion_matrix(list4, pred)
print('Confusion Matrix:')
print(results)

print('Accuracy Score :',accuracy_score(list4, pred))
