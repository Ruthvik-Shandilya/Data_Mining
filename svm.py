import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score

style.use('ggplot')

class SVM:
    def __init__(self, plot_graph=True):
        self.plot_graph = plot_graph
        self.colors = {1: 'r', -1: 'b'}
        if self.plot_graph:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # train
    def fit(self, training_data):
        self.train_data = training_data
        # { ||w||: [w,b] }
        optimum_values = {}
        # Assuming all the possible transforms for the 2d data (To find which quadrant the data lies)
        transforms = [[1, 1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        feature_set = []
        for yi in self.train_data:

            for featureset in self.train_data[yi]:

                for feature in featureset:
                    feature_set.append(feature)

        self.max_feature_value = max(feature_set)
        self.min_feature_value = min(feature_set)
        feature_set = None

        #Asssuming step sizes based on Max Feature value;
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,]

        b_range_multiple = 1
        b_multiple = 3
        w_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([w_optimum, w_optimum])

            optimized = False
            while not optimized:
                #Iterating for all combinations of b
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):

                    for transformation in transforms:
                        # Multiply with assumed value of w with each transform
                        w_t = w * transformation
                        found_option = True

                        for i in self.train_data:
                            for xi in self.train_data[i]:
                                yi = i

                                if yi * (np.dot(w_t, xi) + b) < 1:
                                    # If any feature value's dot product is less than 1, then ignore that weight matrix
                                    found_option = False
                                break

                        if found_option:
                            optimum_values[np.linalg.norm(w_t)] = [w_t, b]
                        break

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in optimum_values])
            # ||w|| : [w,b]
            optimum = optimum_values[norms[0]]

            self.w = optimum[0]
            self.b = optimum[1]

            w_optimum = optimum[0][0] + step * 2


    def predict(self, testing_data):
        # sign( x.w+b )
        results = np.sign(np.dot(np.array(testing_data), self.w) + self.b)
        if results != 0 and self.plot_graph:
            self.ax.scatter(testing_data[0], testing_data[1], s=200, marker='*', c=self.colors[results])
        return results

    def visualize(self):

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        data_range = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]

        # (w.x+b) = 1
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # (w.x+b) = -1
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x+b) = 0
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


header = ['Labels','Feature_0','Product_Name','Feature_1', 'Feature_2']
data_read = pd.read_csv(r"train_dataset.csv",sep=',', usecols=header)
data_read1=data_read[['Labels','Feature_0','Product_Name','Feature_1', 'Feature_2']]

my_dict={}
list_tp=[]
list_tn=[]

for index,row in data_read1.iterrows():
    if row['Labels']==1:
        list_tp.append(row[3:].tolist())
    else:
        list_tn.append(row[3:].tolist())

my_dict[1]=np.asarray(list_tp)
my_dict[-1]=np.asarray(list_tn)

svm = SVM()
svm.fit(training_data=my_dict)


header = ['Labels','Feature_0','Product_Name','Feature_1', 'Feature_2']
data_read2 = pd.read_csv(r"test_dataset.csv",sep=',', usecols=header)
data_read3=data_read2[['Labels','Feature_0','Product_Name','Feature_1', 'Feature_2']]

list_test_features = []
label = data_read3['Labels']
list_test_labels = []

for index,row in data_read3.iterrows():
        list_test_features.append(row[3:].tolist())

for x in label:
        list_test_labels.append(x)

predict_us = list_test_features
pred = []

for p in predict_us:
    pred.append(svm.predict(p))

svm.visualize()

results = confusion_matrix(list_test_labels, pred, labels=[1, -1])
print('Confusion Matrix:')
print(results)

print(classification_report(list_test_labels, pred))
print('Accuracy Score :', accuracy_score(list_test_labels, pred))


products_list = data_read3['Product_Name']
products = []
for x in products_list:
        products.append(x)

complete_list = []
def perf_measure(products,y_actual, y_pred):

    for i in range(len(y_pred)):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        if y_actual[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]==-1:
           TN += 1
        if y_pred[i]==-1 and y_actual[i]!=y_pred[i]:
           FN += 1

        complete_list.append([products[i],y_actual[i],y_pred[i],TP, FP, TN, FN])

perf_measure(products, list_test_labels, pred)
header = ['Products_Name','Actual','Predicted','TP','FP','TN','FN']
dfObj = pd.DataFrame(complete_list,columns=header)
dfObj.to_csv("Output_SVM.csv",index=False)