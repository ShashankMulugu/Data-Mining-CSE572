import train
import pandas as pd
import numpy as np
import collections
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


test_dataset = pd.read_csv("C:\\Users\\Shashank\\Documents\\IMP\\ASU\\Courses\\CSE 572\\Project 3\\proj3_test.csv", header=None)
# test_label = test_dataset['label']
# test_label = test_dataset.drop(['label'])
test_dataset = test_dataset.dropna(how="all").reset_index(drop=True).interpolate()

rms, auc, dt, rr, dc, moving_average, mamax, mamin, fft_max, fft_min, fft_vals = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)
for i in range(len(test_dataset.index)):
    dc_row = []
    rms_row = np.sqrt(np.mean(test_dataset.iloc[i, 0:30] ** 2))
    rms.append(rms_row)
    moving_average = moving_avg(test_dataset.iloc[i, 0:30].tolist(), 5).tolist()
    moving_average = pad(moving_average, 28, np.nan)
    auc_row = abs(simps(test_dataset.iloc[i, 0:30], dx=1))
    auc.append(auc_row)
    sum = 0
    for j in range(29):
        difference = abs(test_dataset.iloc[i, j] - test_dataset.iloc[i, j + 1])
        sum += difference
        dc_row.append(difference)
    dt.append(sum)
    rr_row = sum ** 2 / auc_row
    rr.append(rr_row)
    hist, _ = np.histogram(dc_row, bins=5, range=(0, 15), density=True)
    dc.append(list(hist))
    mamax.append(max(moving_average))
    mamin.append(min(moving_average))


test_dataset = pd.DataFrame()
test_dataset["RMS"] = rms
test_dataset["AUC"] = auc
test_dataset["DT"] = dt
test_dataset["RR"] = rr
test_dataset["MAMAX"] = mamax
test_dataset["MAMIN"] = mamin


mini, maxi, varr = [], [], []
rff = rfft(test_dataset)
for i in range(len(rff)):
    m = min(rff[i])
    ma = max(rff[i])
    variance = np.var(rff[i])
    mini.append(m)
    maxi.append(ma)
    varr.append(variance)


test_dataset["FFTMAX"] = mini
test_dataset["FFTMIN"] = maxi
test_dataset["FFTVAR"] = varr

imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
test_dataset = imp_mean.fit_transform(test_dataset)
test_dataset = StandardScaler().fit_transform(test_dataset)
pca = PCA(n_components=5)
test_dataset = pca.fit_transform(test_dataset)

test_dataset = pd.DataFrame(test_dataset)

loaded_model = pickle.load(open("knn_classifier", "rb"))

predictions = loaded_model.predict(test_dataset)
print('KMeans Predictions',predictions)

dbscan_loaded_model = pickle.load(open("dbscan_knn_classifier", "rb"))

dbscan_predictions = dbscan_loaded_model.predict(test_dataset)
print('DBSCAN Predictions',dbscan_predictions)

print("Kmeans", collections.Counter(predictions))
print("DBscan", collections.Counter(dbscan_predictions))
labels = pd.DataFrame()

labels = labels.assign(Dbscan = dbscan_predictions)
labels = labels.assign(Kmeans = predictions)

labels.to_csv('labels.csv', header=False, index=False)