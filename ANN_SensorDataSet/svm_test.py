from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from time import time
import numpy as np
import datetime
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import label_binarize

def load_excel(path):
    data = pd.read_excel(path, engine='openpyxl')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

path = 'C:/Users/91702/OneDrive/Desktop/tmp/EMG_CNN-main/ANN_SensorDataSet/bu_data_for_ML.xlsx'
X, y = load_excel(path)

print("Initial shapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)

X = StandardScaler().fit_transform(X)
print("Transformed X shape:", X.shape)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
kernel = "rbf"

time0 = time()
clf = SVC(kernel=kernel, gamma="auto", degree=1, cache_size=7000, decision_function_shape='ovr').fit(Xtrain, Ytrain)
print("Accuracy under kernel '%s': %f" % (kernel, clf.score(Xtest, Ytest)))
print("Time taken: ", datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

gamma_range = np.logspace(-10, 1, 50)
score = []
for i in gamma_range:
    clf = SVC(kernel="rbf", gamma=i, cache_size=5000, decision_function_shape='ovr').fit(Xtrain, Ytrain)
    score.append(clf.score(Xtest, Ytest))
best_gamma = gamma_range[score.index(max(score))]
print("Best gamma:", best_gamma)
print("Best score:", max(score))
print(classification_report(Ytest, clf.predict(Xtest)))

def print_roc(Ytest, Xtest, clf):
    y_bin = label_binarize(Ytest, classes=np.unique(Ytest))
    y_score = clf.decision_function(Xtest)
    
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(y_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10,10))
    for i in range(y_bin.shape[1]):
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class %d (area = %0.2f)' % (i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

print_roc(Ytest, Xtest, clf)

