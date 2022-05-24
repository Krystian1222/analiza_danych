# Klasyfikacja wieloklasowa

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

# Zadanie 1 - Wygenerowanie przykładowych danych z liczbą klas równą cztery, ustalenie liczby atrybutów na dwa.
X, y = make_classification(
    n_classes = 4, n_clusters_per_class = 1, n_samples = 2000,
    n_informative = 2, n_redundant = 0, n_features = 2, random_state = 5
)
plt.figure(figsize = (8, 6))
plt.title("Zadanie 1. Wizualizacja przykładowych danych.")
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.savefig("Zadanie_1_wizualizacja.png")

# Zadanie 2 - Podzielenie danych na część uczącą i testującą w proporcjach 50/50.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 200)

# Zadanie 3 - Utworzenie listy 4 par klasyfikatorów w taki sposób, aby klasyfikator był opakowany w funkcji OneVsOneClassifier() i OneVsRestClassifier().
metody = [OVOC, OVRC]
klasyfikatory = [SVC(kernel = 'linear', probability = True),
                 SVC(kernel = 'rbf', probability = True),
                 LogReg(), Perceptron()]

df = pd.DataFrame(columns = ['accuracy_score',
                  'recall_score', 'precision_score_tab',
                  'f1_score', 'roc_auc_score'])

nazwy = ['OVOC SVC linear', 'OVOC SVC rbf', 'OVOC LogisticRegression', 'OVOC Perceptron',
         'OVRC SVC linear', 'OVRC SVC rbf', 'OVRC LogisticRegression', 'OVRC Perceptron']

# Zadanie 4 - Dla każdego klasyfikatora z listy i strategii OvO i OvR wykonano kolejne polecenia:
# - wykonano uczenie na zbiorze uczącym, a następnie wyznaczono predykcję na zbiorze testowym,
# - zwizualizowano wyniki klasyfikacji: poprawne i błędne,
# - wyznaczono następujące miary jakości klasyfikacji:
#   - dokładność,
#   - czułość,
#   - precyzję,
#   - F1,
#   - polę pod krzywą ROC
# - wyznaczono i narysowano krzywą ROC,
# - narysowano powierzchnię dyskryminacyjną.
licz = 0
for i in metody:
    for j in klasyfikatory:
        licz += 1
        acc_score_tab = []
        rec_score_tab = []
        pre_score_tab = []
        f1_score_tab = []
        roc_auc_score_tab = []
        clf = i(j).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        plt.figure(figsize=(10, 5))
        plt.subplot(131)
        plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
        plt.title("oczekiwane")
        plt.subplot(132)
        plt.scatter(X_test[:, 0], X_test[:, 1], c = y_pred)
        plt.title("obliczone")
        plt.subplot(133)
        plt.title("różnice")
        for j in range(y_test.shape[0]):
            if(y_test[j] != y_pred[j]):
                plt.scatter(X_test[j, 0], X_test[j, 1], c = 'r')
            else:
                plt.scatter(X_test[j, 0], X_test[j, 1], c = 'g')
        plt.savefig("Zadanie_4_WK_{}".format(nazwy[licz - 1]))
        acc_score_tab.append(accuracy_score(y_test, y_pred))
        rec_score_tab.append(recall_score(y_test, y_pred, average = 'micro'))
        pre_score_tab.append(precision_score(y_test, y_pred, average = 'micro'))
        f1_score_tab.append(f1_score(y_test, y_pred, average = 'micro'))
        LB = LabelBinarizer()
        LB.fit(y_test)
        y_true = LB.transform(y_test)
        y_score = LB.transform(y_pred)
        if i == OVOC:
            roc_auc_score_tab.append(roc_auc_score(y_true, y_score, average = 'micro', multi_class = 'ovo'))
        else:
            roc_auc_score_tab.append(roc_auc_score(y_true, y_score, average = 'micro', multi_class = 'ovr'))
        df = df.append({'accuracy_score': np.mean(acc_score_tab),
                        'recall_score': np.mean(rec_score_tab),
                        'precision_score_tab': np.mean(pre_score_tab),
                        'f1_score': np.mean(f1_score_tab),
                        'roc_auc_score': np.mean(roc_auc_score_tab)},
                        ignore_index = True)

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
        plt.title('Krzywa ROC dla {}'.format(nazwy[licz - 1]))
        for l in range(n_classes):
            lw = 2
            plt.plot(fpr[l], tpr[l], lw = lw, label = 'krzywa ROC (AUC = %0.2f dla klasy ' % roc_auc[l] + str(l) + ')')
            plt.plot([0, 1], [0, 1], 'k--')
        plt.legend(loc='lower right')
        plt.savefig("Zadanie_4_ROC_{}.png".format(nazwy[licz - 1]))

        plt.figure()
        x = np.arange(-5, 5, 0.1)
        y = np.arange(-5, 5, 0.1)
        xx, yy = np.meshgrid(x, y, sparse=False)
        Trnsp = np.vstack(([xx.T], [yy.T])).T
        Trnsp_r = Trnsp.reshape(10000, 2)
        Trnsp_pred = clf.predict(Trnsp_r)
        plt.contourf(x, y, Trnsp_pred.reshape((100, 100)), edgecolor='none', alpha=0.5,
                     cmap=plt.cm.get_cmap('Spectral', 4))
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('Spectral', 4))
        plt.title("Krzywa dyskryminacyjna dla {}".format(nazwy[licz - 1]))
        plt.savefig("Zadanie_4_KD_{}.png".format(nazwy[licz - 1]))


# 5 - Zwizualizowano otrzymane wyniki dotyczące jakości klasyfikacji.
print(df)
df_np = df.to_numpy()
print(df_np)
print(df_np[0])
plt.figure(figsize = (8, 6))
licznik = 0
n_groups = 5
index = np.arange(n_groups)
bar_width = 0.1
labelsy = ['OVO=SVC linear', 'OVO=SVC rbf', 'OVO=LogisticR', 'OVO=Perceptron',
           'OVR==SVC linear', 'OVR==SVC rbf', 'OVR==LogisticR', 'OVR==Perceptron']
rects = []
for i in range(8):
    rects.append(plt.bar(index + (i + 1) * bar_width, df_np[i], bar_width, label = labelsy[i]))
plt.xticks(index + 3 * bar_width, ('accuracy_score',
'recall_score', 'precision_score', 'f1_score', 'roc_auc'),
           rotation = 'vertical')
plt.legend()
plt.tight_layout()
plt.savefig("Zadanie_5_porownanie.png")