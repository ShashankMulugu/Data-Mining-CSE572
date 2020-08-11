# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:51:03 2020

@author: Shashank
"""

import numpy as np
import pandas as pd 
import pickle
import collections
from matplotlib import pyplot as plt
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
from sklearn.neighbors import NearestNeighbors
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

list_of_classifiers = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Poly SVM", "Gaussian Process",
                "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                "Naive Bayes", "QDA","XGBoost", "SGD Classifier","Histogram Boosting Cls", "Bagging Cls"]

classifiers = [ KNeighborsClassifier(5),
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
                XGBClassifier(), SGDClassifier(loss="hinge", penalty="l2", max_iter=100), HistGradientBoostingClassifier(max_iter=100), BaggingClassifier(GaussianNB(),max_samples=0.5, max_features=0.5)]

accuracy_arr = []
def k_fold(X, y, list_of_classifiers, classifiers):
    kf = KFold(n_splits = 10, shuffle=True)
    for classifier_name, classifier_fn_call in zip(list_of_classifiers, classifiers):
            accuracy, f1, precision, recall = [], [], [], []
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
            accuracy_arr.append(np.mean(accuracy))
            print('Accuracy metrics for ' +classifier_name)
            print('Accuracy: ', np.mean(accuracy))
            print('F1 score: ', np.mean(f1))
            print('Precison: ', np.mean(precision))
            print('Recall: ', np.mean(recall))
                        
            
            
def moving_avg(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

def pad(seq, target_len, padding = None):
    length = len(seq)
    if length > target_len:
        print("Too long for I/P")
        
    seq.extend([padding] * (target_len - length))
    return seq

dataset_meal = pd.DataFrame()
dataset_amount = pd.DataFrame()
for i in range(5):
        df = pd.read_csv("C:\\Users\\Shashank\\Documents\\IMP\\ASU\\Courses\\CSE 572\\Project 3\\mealData"+"{}.csv".format(i+1), names = [i for i in range(31)], header = None, nrows = 50) 
        df_amount = pd.read_csv("C:\\Users\\Shashank\\Documents\\IMP\\ASU\\Courses\\CSE 572\\Project 3\\mealAmountData"+"{}.csv".format(i+1), header = None, nrows = 50) 
        dataset_meal = dataset_meal.append(df, ignore_index = True)
        dataset_amount = dataset_amount.append(df_amount, ignore_index = True)

# dataset_meal = dataset_meal.dropna(how='all') 
dataset_meal = dataset_meal.interpolate(method ='linear', limit_direction ='both') 
dataset_meal = dataset_meal.drop([30],axis = 1)

dataset_amount = dataset_amount.dropna(how='all') 

dataset = dataset_meal

rms_val, auc_val, dt_val, rr_val, dc, moving_average, mamax, mamin, fft_max, fft_min, fft_vals = [], [], [], [], [], [], [], [], [], [], []
#extract features
for i in range(len(dataset.index)):
    dc_row=[]
    rms_row = np.sqrt(np.mean(dataset.iloc[i, 0:30]**2))
    rms_val.append(rms_row)
    moving_average = moving_avg(dataset.iloc[i,0:30].tolist(), 5).tolist()
    moving_average = pad(moving_average, 28, np.nan)
    auc_row = abs(simps(dataset.iloc[i, 0:30], dx=1))
    auc_val.append(auc_row)
    sum = 0
    for j in range(29):
            difference =  abs(dataset.iloc[i,j] - dataset.iloc[i,j+1])
            sum += difference
            dc_row.append(difference)
    dt_val.append(sum)
    rr_row = sum**2 /auc_row
    rr_val.append(rr_row)
    hist, _ = np.histogram(dc_row, bins=5, range=(0,15), density=True)
    dc.append(list(hist))
    mamax.append(max(moving_average))
    mamin.append(min(moving_average))
    
    
dataset = pd.DataFrame()
dataset['RMS'] = rms_val
dataset['AUC'] = auc_val
dataset['DT'] = dt_val
dataset['rr_val'] = rr_val
dataset['MAMAX'] = mamax
dataset['MAMIN'] = mamin


fft_min, fft_max, fft_varr = [], [], []
rff = rfft(dataset)
for i in range(len(rff)):
    m = min(rff[i])
    ma = max(rff[i])
    variance = np.var(rff[i])
    fft_min.append(m)
    fft_max.append(ma)
    fft_varr.append(variance)


dataset['FFTMAX'] = fft_max
dataset['FFTMIN'] = fft_min
dataset['FFTVAR'] = fft_varr

dataset = StandardScaler().fit_transform(dataset)
pca = PCA(n_components=5)
dataset = pca.fit_transform(dataset)

dataset = pd.DataFrame(dataset)

print("Kmeans Result")
km_clusters = KMeans(n_clusters=6, random_state=12).fit(dataset)
print(km_clusters.labels_)
print(collections.Counter(km_clusters.labels_))
print("Kmeans SSE:",km_clusters.inertia_)

clustering_dict = {i: np.where(km_clusters.labels_ == i)[0] for i in range(km_clusters.n_clusters)}

ground_truth_array = dataset_amount.to_numpy().reshape(-1)
ground_truth = []
for i in range(len(ground_truth_array)):
    if ground_truth_array[i] == 0:
        ground_truth.append(1)
    elif ground_truth_array[i] > 0 and ground_truth_array[i] <= 20:
        ground_truth.append(2)
    elif ground_truth_array[i] >= 21 and ground_truth_array[i] <= 40:
        ground_truth.append(3)
    elif ground_truth_array[i] >= 41 and ground_truth_array[i] <= 60:
        ground_truth.append(4)
    elif ground_truth_array[i] >= 61 and ground_truth_array[i] <= 80:
        ground_truth.append(5)
    elif ground_truth_array[i] >= 81 and ground_truth_array[i] <= 100:
        ground_truth.append(6)
    else:
        ground_truth.append(6)
print("ground truth result:", ground_truth)
print(collections.Counter(ground_truth))
# ground_truth_dict = dict(enumerate(ground_truth))
# print(ground_truth_dict)

# print(clustering_dict)

cluster_bin_assignment = []
for i in clustering_dict.keys():
    cluster_arr = clustering_dict[i]
    bin1 = 0
    bin2 = 0
    bin3 = 0
    bin4 = 0
    bin5 = 0
    bin6 = 0
    for i in range(len(cluster_arr)):
        if ground_truth[i] == 1:
            bin1+=1
        elif ground_truth[i] == 2:
            bin2+=1
        elif ground_truth[i] == 3:
            bin3+=1
        elif ground_truth[i] == 4:
            bin4+=1
        elif ground_truth[i] == 5:
            bin5+=1
        elif ground_truth[i] == 6:
            bin6+=1
    bin_dict = {bin1:1,bin2:2,bin3:3,bin4:4,bin5:5,bin6:6}
    binval = bin_dict.get(max(bin_dict))
    cluster_bin_assignment.append(binval)

print("Cluster Equivalents")
print(cluster_bin_assignment)

cluster_bin1_6 = km_clusters.labels_

new_cluster_bins = []
for i in range(len(cluster_bin1_6)):
    new_cluster_bins.append(cluster_bin_assignment[cluster_bin1_6[i]])


knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(dataset, new_cluster_bins)
pickle_name = "knn_classifier"
pickle.dump(knn, open(pickle_name, 'wb'))

# DBSCAN Clustering:-
dbscan_clusters = DBSCAN(eps=0.75, min_samples=3).fit(dataset)
print("DBSCAN Clustering results")
print(dbscan_clusters.labels_)
print(collections.Counter(dbscan_clusters.labels_))
dbscan_array = dbscan_clusters.labels_

dbscan_clustering_dict = {i: np.where(dbscan_clusters.labels_ == i)[0] for i in range(len(collections.Counter(dbscan_array).keys()))}
print(dbscan_clustering_dict)

dbscan_bin_assignment = []
for i in dbscan_clustering_dict.keys():
    cluster_arr = dbscan_clustering_dict[i]
    bin1 = 0
    bin2 = 0
    bin3 = 0
    bin4 = 0
    bin5 = 0
    bin6 = 0
    for i in range(len(cluster_arr)):
        if ground_truth[i] == 1:
            bin1+=1
        elif ground_truth[i] == 2:
            bin2+=1
        elif ground_truth[i] == 3:
            bin3+=1
        elif ground_truth[i] == 4:
            bin4+=1
        elif ground_truth[i] == 5:
            bin5+=1
        elif ground_truth[i] == 6:
            bin6+=1
    bin_dict = {bin1:1,bin2:2,bin3:3,bin4:4,bin5:5,bin6:6}
    binval = bin_dict.get(max(bin_dict))
    dbscan_bin_assignment.append(binval)

print("DBSCAN Bin  Equivalents")
print(dbscan_bin_assignment)


dbscan_cluster_bin1_6 = dbscan_clusters.labels_

dbscan_new_cluster_bins = []
for i in range(len(dbscan_cluster_bin1_6)):
    dbscan_new_cluster_bins.append(dbscan_bin_assignment[dbscan_cluster_bin1_6[i]])

print(dbscan_new_cluster_bins)

dbscan_knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
dbscan_knn.fit(dataset, dbscan_new_cluster_bins)
pickle_name = "dbscan_knn_classifier"
pickle.dump(dbscan_knn, open(pickle_name, 'wb'))

# Figuring out the optimal Epsilon Value using NN
# neigh = NearestNeighbors(n_neighbors=2)
# nbrs = neigh.fit(dataset)
# distances, indices = nbrs.kneighbors(dataset)

# distances = np.sort(distances, axis=0)
# distances = distances[:,1]
# plt.plot(distances)
# plt.show()





 


