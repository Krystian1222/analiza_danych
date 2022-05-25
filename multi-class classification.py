# Multi-class classification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsOneClassifier as OVOC
from sklearn.multiclass import OneVsRestClassifier as OVRC
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize

# Exercise 1 - Generate sample data with the number of classes equal to four, setting the number of attributes to two.
X, y = make_classification(
    n_classes=4, n_clusters_per_class=1, n_samples=2000, n_informative=2, n_redundant=0, n_features=2, random_state=5
)

plt.figure(figsize=(8, 6))
plt.title('Exercise 1. Visualisation of sample data.')
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig('Exercise_1_visualisation.png')

# Exercise 2 - Split the data into a 50/50 training and testing component.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=200)

# Exercise 3 - Create a list of 4 pairs of classifiers such that the classifier is wrapped in OneVsOneClassifier() and OneVsRestClassifier() functions.
methods = [OVOC, OVRC]
classifiers = [SVC(kernel='linear', probability=True),
               SVC(kernel='rbf', probability=True),
               LogReg(),
               Perceptron()]

df = pd.DataFrame(columns=['accuracy_score', 'recall_score', 'precision_score_tab', 'f1_score', 'roc_auc_score'])

names = ['OVOC SVC linear', 'OVOC SVC rbf', 'OVOC LogisticRegression', 'OVOC Perceptron', 'OVRC SVC linear', 'OVRC SVC rbf', 'OVRC LogisticRegression', 'OVRC Perceptron']

# Exercise 4 - For each classifier in the list and the OvO and OvR strategies, the following instructions were performed:
# - performed learning on the training set and then determined the prediction on the test set,
# - classification results were visualised: correct and incorrect,
# - the following measures of classification quality were determined:
#      - accuracy,
#      - sensitivity,
#      - precision,
#      - F1,
#      - area under the ROC Curve,
# - ROC Curve was determined and drawn,
# - discriminant surface was drawn.
count = 0
for i in methods:
    for j in classifiers:
        count += 1
        acc_score_tab = []
        rec_score_tab = []
        pre_score_tab = []
        f1_score_tab = []
        roc_auc_score_tab = []
        clf = i(j).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        plt.figure(figsize=(10, 5))
        plt.subplot(131)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        plt.title('expected')
        plt.subplot(132)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        plt.title('calculated')
        plt.subplot(133)
        plt.title('differences')
        for k in range(y_test.shape[0]):
            if y_test[k] != y_pred[k]:
                plt.scatter(X_test[k, 0], X_test[k, 1], c='r')
            else:
                plt.scatter(X_test[k, 0], X_test[k, 1], c='g')
        plt.savefig('Exercise_4_WK_{}'.format(names[count - 1]))
        acc_score_tab.append(accuracy_score(y_test, y_pred))
        rec_score_tab.append(recall_score(y_test, y_pred, average='micro'))
        pre_score_tab.append(precision_score(y_test, y_pred, average='micro'))
        f1_score_tab.append(f1_score(y_test, y_pred, average='micro'))
        LB = LabelBinarizer()
        LB.fit(y_test)
        y_true = LB.transform(y_test)
        y_score = LB.transform(y_pred)
        if i == OVOC:
            roc_auc_score_tab.append(roc_auc_score(y_true, y_score, average = 'micro', multi_class='ovo'))
        else:
            roc_auc_score_tab.append(roc_auc_score(y_true, y_score, average = 'micro', multi_class='ovr'))
        df = df.append({'accuracy_score': np.mean(acc_score_tab),
                        'recall_score': np.mean(rec_score_tab),
                        'precision_score_tab': np.mean(pre_score_tab),
                        'f1_score': np.mean(f1_score_tab),
                        'roc_auc_score': np.mean(roc_auc_score_tab)},
                        ignore_index=True)

        plt.figure()
        Y_score = clf.fit(X_train, y_train).decision_function(X_test)
        Y_test = label_binarize(y_test, classes = [0, 1, 2, 3])
        n_classes = Y_test.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for k in range(n_classes):
            fpr[k], tpr[k], _ = metrics.roc_curve(Y_test[:, k], Y_score[:, k])
            roc_auc[k] = metrics.auc(fpr[k], tpr[k])
        fpr['micro'], tpr['micro'], _ = metrics.roc_curve(Y_test.ravel(), Y_score.ravel())
        roc_auc['micro'] = metrics.auc(fpr['micro'], tpr['micro'])
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Roc Curve for {}'.format(names[count - 1]))
        for l in range(n_classes):
            lw = 2
            plt.plot(fpr[l], tpr[l], lw = lw, label = 'ROC Curve (AUC = %0.2f for class ' % roc_auc[l] + str(l) + ')')
            plt.plot([0, 1], [0, 1], 'k--')
        plt.legend(loc='lower right')
        plt.savefig("Exercise_4_ROC_{}.png".format(names[count - 1]))

        plt.figure()
        x = np.arange(-5, 5, 0.1)
        y = np.arange(-5, 5, 0.1)
        xx, yy = np.meshgrid(x, y, sparse=False)
        Trnsp = np.vstack(([xx.T], [yy.T])).T
        Trnsp_r = Trnsp.reshape(10000, 2)
        Trnsp_pred = clf.predict(Trnsp_r)
        plt.contourf(x, y, Trnsp_pred.reshape((100, 100)), edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 4))
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 4))
        plt.title("Discrimination curve for {}".format(names[count - 1]))
        plt.savefig("Exercise_4_DC_{}.png".format(names[count - 1]))

# 5 - The results obtained regarding the quality of the classification were visualised.
print(df)
df_np = df.to_numpy()
print(df_np)
print(df_np[0])
plt.figure(figsize = (8, 6))
n_groups = 5
index = np.arange(n_groups)
bar_width = 0.1
labels = ['OVO=SVC linear', 'OVO=SVC rbf', 'OVO=LogisticR', 'OVO=Perceptron',
           'OVR==SVC linear', 'OVR==SVC rbf', 'OVR==LogisticR', 'OVR==Perceptron']
rects = []
for i in range(8):
    rects.append(plt.bar(index + (i + 1) * bar_width, df_np[i], bar_width, label = labels[i]))
plt.xticks(index + 3 * bar_width, ('accuracy_score',
'recall_score', 'precision_score', 'f1_score', 'roc_auc'),
           rotation='vertical')
plt.legend()
plt.tight_layout()
plt.savefig("Exercise_5_comparision.png")