# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:24:11 2020

@author: Shashank
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:52:16 2020

@author: Shashank
"""
import numpy as np
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot as plt
from scipy.fftpack import fft,rfft
import math
import statistics
from scipy.signal import welch
from sklearn.decomposition import PCA
from itertools import *
from sklearn.impute import SimpleImputer
import plotly.express as px
import seaborn as sns

def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = rfft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

def moving_avg(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

def pad(seq, target_len, padding = None):
    length = len(seq)
    if length > target_len:
        print(length)
        raise TooLongError("sequence too long ({}) for target length {}"
                           .format(length, target_length))
        
    seq.extend([padding] * (target_len - length))
    return seq



cgm_ts4 = pd.read_csv("C:\\Users\\Shashank\\Documents\\IMP\\ASU\\Courses\\CSE 572\\DataFolder\\CGMDatenumLunchPat4.csv")
cgm4 = pd.read_csv("C:\\Users\\Shashank\\Documents\\IMP\\ASU\\Courses\\CSE 572\\DataFolder\\CGMSeriesLunchPat4.csv")




cgm4 = cgm4.dropna(how='all').reset_index(drop=True).interpolate()
cgm_ts4 = cgm_ts4.dropna(how='all').reset_index(drop=True).interpolate()

# for i in range(0, len(cgm4.index)):
#     for j in range(0, len(cgm4.columns)):
#         print (cgm4.values[i,j],cgm_ts3.values[i,j],i,j)


#Keep
# means = []
# maxs = []
# stdd = []

# for i in range(33):
#     means.append(cgm.loc[i,:].mean())
#     maxs.append(cgm.loc[i,:].max())
#     stdd.append(cgm.loc[i,:].std())

#---------------------------------------
#Keep   


feature = []
fl = []
il = []
for i in range(len(cgm4.values)):
    feature_row = []
    # Feature Type 1 - Rolling Window 
    moving_average = moving_avg(cgm4.loc[i,:].tolist(), 5).tolist()
    moving_average = pad(moving_average, 38, np.nan)
    feature_row.extend(moving_average)
    # plt.figure(i)
    # plt.plot(moving_avg(cgm_ts4.loc[i,:].tolist(), 5), moving_avg(cgm2.loc[i,:].tolist(), 5))
    # plt.plot(cgm_ts4.loc[i,:].tolist(), cgm2.loc[i,:].tolist())
    
    il.append(len(moving_average))
    
    # Feature Type 2 - Analyzing in the Frequency Domain - FFT & PSD
    y_values = cgm4.loc[i,:].dropna().unique().tolist()
    y_values_ts = cgm_ts4.loc[i,:].dropna().unique().tolist()
    num = len(y_values)
    t_n = y_values[num-1] - y_values[0]
    T = t_n / num
    fs = 1 / T
    f_values, fft_values = get_fft_values(y_values, T, num, fs)
    # plt.figure(i+1)
    # plt.plot(f_values, fft_values, linestyle = '-', color = 'blue')
    # plt.xlabel('Frequency [Hz]', fontsize=16)   
    # plt.ylabel('Amplitude', fontsize=16)
    fft_values = pad(fft_values.tolist(), 20, np.nan)
    feature_row.extend(fft_values)
    il.append(len(fft_values))    
    
    # PSD (Power Spectral Density)
    psdf_values, psd_values = get_psd_values(y_values, T, num, fs)
    
    # plt.figure(i+2)
    # plt.plot(psdf_values, psd_values, linestyle='-', color='blue')
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('PSD [V**2 / Hz]')
    
    psd_values = pad(psd_values.tolist(), 20, np.nan)
    feature_row.extend(psd_values)
    il.append(len(psd_values))
    
    # #Feature Type 3 - Statistical Methods
    feature_row.extend([statistics.stdev(cgm4.loc[i,:])])
    
    feature_row.extend([max(cgm4.loc[i,:])])
    
    feature_row.append(np.sqrt(np.mean(cgm4.loc[i,:]**2)))
    
    # Feature Type 4 - Rate of icrease of CGM levels per unit time (or) Velocity
    max_slope = [x - z for x, z in zip(y_values[:-1], y_values[1:])]
    max_slope_time = [x - z for x, z in zip(y_values_ts[:-1], y_values_ts[1:])]
    velocity = [y / w for y, w in zip(max_slope, max_slope_time)]
    velocity_zero_crossings = np.where(np.diff(np.signbit(velocity)))[0]
    velocity = pad(velocity, 37, np.nan)
    velocity_zero_crossings = pad(velocity_zero_crossings.tolist(), 20, np.nan)
    velocity_mean = np.mean(velocity)
    velocity_max = np.max(velocity)
    # plt.figure(i+3)
    # plt.plot(y_values_ts, pad(velocity, int(len(y_values_ts)), 0))
    # plt.xlabel('time')
    # plt.ylabel('velocity')
    feature_row.extend(velocity)
    feature_row.extend(velocity_zero_crossings)
    feature_row.append(velocity_mean)
    feature_row.append(velocity_max)
    feature.append(feature_row)
    fl.append(len(feature_row))
    il.append(len([y / w for y, w in zip(max_slope, max_slope_time)]))
    
print(feature)


feature_2D_arr = np.asarray(feature).reshape(len(feature),140)
print(feature_2D_arr, feature_2D_arr.shape)


# Replacing missing values with imputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
feature_ip = imp_mean.fit_transform(feature_2D_arr)

pca = PCA(n_components=5)
principalComps = pca.fit_transform(feature_ip)
print(pca.explained_variance_ratio_)
components = abs(pca.components_)
variances = pca.explained_variance_
x = [i for i in range(0, len(components[0]))]
for i in range(0, 5):
    plt.figure(figsize=(100,70))
    plt.bar(x, components[i])
    plt.xticks(np.arange(len(components[0])), x, rotation=90)    
    plt.show()
    positives = np.array(np.argwhere(components[i] > 0).flatten())
    positive_sorted = np.argsort(components[i][:])
    positives = positive_sorted
#     print(components[i])
#     print(positives)
#     print(columns)
    print(variances[i])
    #print(columns[positives])
    print(components[i][positives])
    print(components[i])

# Radar plot with all the features
radarPlotPCA = pd.DataFrame(dict(
    r=pca.explained_variance_ratio_,
    theta=['PC1','PC2','PC3',
            'PC4', 'PC5']))

fig = px.line_polar(radarPlotPCA, r='r', theta='theta', line_close=True)
fig.show()
fig.write_image("C:\\Users\\Shashank\\Documents\\IMP\\ASU\\Courses\\CSE 572\\DataFolder\\fig4.png")
