# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:51:03 2020

@author: Shashank
"""

import numpy as np
import pandas as pd
import pickle
from scipy.fftpack import rfft
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.integrate import simps
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

list_of_classifiers = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Poly SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "XGBoost",
    "SGD Classifier",
    "Histogram Boosting Cls",
    "Bagging Cls",
]

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    SVC(kernel="poly"),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    XGBClassifier(),
    SGDClassifier(loss="hinge", penalty="l2", max_iter=100),
    HistGradientBoostingClassifier(max_iter=100),
    BaggingClassifier(GaussianNB(), max_samples=0.5, max_features=0.5),
]

accuracy_arr = []
fscore_arr = []
roc_score_arr = []
recall_arr = []


def k_fold(X, y, list_of_classifiers, classifiers):
    kf = KFold(n_splits=10, shuffle=True)
    for classifier_name, classifier_fn_call in zip(list_of_classifiers, classifiers):
        accuracy, f1, precision, recall, roc_score = [], [], [], [], []
        for train_index, test_index in kf.split(X):
            train_X, test_X = X.iloc[train_index], X.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            clf = classifier_fn_call
            clf.fit(train_X, ytrain)
            predicted = clf.predict(test_X)
            accuracy.append(metrics.accuracy_score(ytest, predicted))
            f1.append(metrics.f1_score(ytest, predicted))
            precision.append(metrics.precision_score(ytest, predicted))
            recall.append(metrics.recall_score(ytest, predicted))
            roc_score.append(metrics.roc_auc_score(ytest, predicted))
            cf_matrix = metrics.confusion_matrix(ytest, predicted)
        accuracy_arr.append(np.mean(accuracy))
        fscore_arr.append(np.mean(f1))
        roc_score_arr.append(np.mean(roc_score))
        recall_arr.append(np.mean(recall))
        print("Accuracy metrics for " + classifier_name)
        print("Accuracy: ", np.mean(accuracy))
        print("F1 score: ", np.mean(f1))
        print("Precison: ", np.mean(precision))
        print("Recall: ", np.mean(recall))
        print("AUC_ROC: ", np.mean(roc_score))
        print(cf_matrix)


def moving_avg(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def pad(seq, target_len, padding=None):
    length = len(seq)
    if length > target_len:
        raise TooLongError(
            "sequence too long ({}) for target length {}".format(length, target_length)
        )

    seq.extend([padding] * (target_len - length))
    return seq


dataset_meal = pd.DataFrame()
for i in range(5):
    df = pd.read_csv(
        "mealData" + "{}.csv".format(i + 1), names=[i for i in range(31)], header=None
    )
    dataset_meal = dataset_meal.append(df, ignore_index=True)

dataset_meal = dataset_meal.dropna(how="all")
dataset_meal = dataset_meal.interpolate(method="linear", limit_direction="both")
dataset_meal = dataset_meal.drop([30], axis=1)
# print(dataset_meal.head)

dataset_nomeal = pd.DataFrame()
for i in range(5):
    df_nomeal = pd.read_csv(
        "Nomeal" + "{}.csv".format(i + 1), names=[i for i in range(31)], header=None
    )
    dataset_nomeal = dataset_nomeal.append(df_nomeal, ignore_index=True)

dataset_nomeal = dataset_nomeal.dropna(how="all")
dataset_nomeal = dataset_nomeal.interpolate(method="linear", limit_direction="both")

dataset = dataset_meal.append(dataset_nomeal, ignore_index=True)

(
    rms_val,
    auc_val,
    dt_val,
    rr_val,
    dc,
    moving_average,
    mamax,
    mamin,
    fft_max,
    fft_min,
    fft_vals,
) = ([], [], [], [], [], [], [], [], [], [], [])
# extract features
for i in range(len(dataset.index)):
    dc_row = []
    rms_row = np.sqrt(np.mean(dataset.iloc[i, 0:30] ** 2))
    rms_val.append(rms_row)
    moving_average = moving_avg(dataset.iloc[i, 0:30].tolist(), 5).tolist()
    moving_average = pad(moving_average, 28, np.nan)
    auc_row = abs(simps(dataset.iloc[i, 0:30], dx=1))
    auc_val.append(auc_row)
    sum = 0
    for j in range(29):
        difference = abs(dataset.iloc[i, j] - dataset.iloc[i, j + 1])
        sum += difference
        dc_row.append(difference)
    dt_val.append(sum)
    rr_row = sum ** 2 / auc_row
    rr_val.append(rr_row)
    hist, _ = np.histogram(dc_row, bins=5, range=(0, 15), density=True)
    dc.append(list(hist))
    mamax.append(max(moving_average))
    mamin.append(min(moving_average))


dataset = pd.DataFrame()
dataset["RMS"] = rms_val
dataset["AUC"] = auc_val
dataset["DT"] = dt_val
dataset["rr_val"] = rr_val
dataset["MAMAX"] = mamax
dataset["MAMIN"] = mamin


fft_min, fft_max, fft_varr = [], [], []
rff = rfft(dataset)
for i in range(len(rff)):
    m = min(rff[i])
    ma = max(rff[i])
    variance = np.var(rff[i])
    fft_min.append(m)
    fft_max.append(ma)
    fft_varr.append(variance)


dataset["FFTMAX"] = fft_max
dataset["FFTMIN"] = fft_min
dataset["FFTVAR"] = fft_varr

dataset = StandardScaler().fit_transform(dataset)
pca = PCA(n_components=5)
dataset = pca.fit_transform(dataset)

dataset = pd.DataFrame(dataset)

dataset["target"] = [1] * len(dataset_meal.index) + [0] * len(dataset_nomeal.index)

dataset = dataset.sample(frac=1).reset_index(drop=True)

X = dataset.iloc[:, 0:5]
y = dataset.target


k_fold(X, y, list_of_classifiers, classifiers)

best_classifier = list_of_classifiers[np.argmax(fscore_arr)]
with open(list_of_classifiers[np.argmax(fscore_arr)], "wb") as file:
    pickle.dump(classifiers[np.argmax(fscore_arr)], file)

# best_classifier = 'AdaBoost'
# with open(list_of_classifiers[8], 'wb') as file:
#   pickle.dump(classifiers[8], file)
