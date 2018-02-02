# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:37:40 2016

@author: Alex
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, classification_report
import matplotlib.pylab as plt
import datetime as dt
import time
import normalizer


def load_snp_returns():
    f = open('table.csv', 'rb').readlines()[1:]
    raw_data = []
    raw_dates = []
    for line in f:
        try:
            open_price = float(line.split(',')[1])
            close_price = float(line.split(',')[4])
            raw_data.append(close_price - open_price)
            raw_dates.append(line.split(',')[0])
        except:
            continue

    return raw_data[::-1], raw_dates[::-1]


def load_snp_close():
    f = open('table.csv', 'rb').readlines()[1:]
    raw_data = []
    raw_dates = []
    for line in f:
        try:
            close_price = float(line.split(',')[4])
            raw_data.append(close_price)
            raw_dates.append(line.split(',')[0])
        except:
            continue

    return raw_data, raw_dates


def split_into_chunks(data, train, predict, step, binary=True, scale=True):
    X, Y = [], []
    for i in range(0, len(data), step):
        try:
            x_i = data[i:i+train]
            y_i = data[i+train+predict]
            
            # Use it only for daily return time series
            if binary:
                if y_i > 0.:
                    y_i = [1., 0.]
                else:
                    y_i = [0., 1.]

                if scale: x_i = (np.array(x_i) - np.mean(x_i)) / np.std(x_i)
                
            else:
                timeseries = np.array(data[i:i+train+predict])
                if scale:
                    half_window = 11
                    timeseries = timeseries-timeseries[half_window]
                    y_i = timeseries[-1]
                    #y_i = (y_i - np.mean(timeseries[:-1])) / np.std(timeseries[:-1])
                    #x_i = (np.array(timeseries[:-1]) - np.mean(timeseries[:-1])) / np.std(timeseries[:-1])
                    x_i = timeseries[:-1]
                else:
                    x_i = timeseries[:-1]
                    y_i = timeseries[-1]
            
        except:
            break

        X.append(x_i)
        Y.append(y_i)

    return X, Y

def split_into_chunks_adaptive(data, ewm, train, predict, step, binary=True, scale=True):
    X, Y, shift = [], [], []
    for i in range(0, len(data), step):
        try:
            x_i = data[i:i+train]
            y_i = data[i+train+predict]
            
            # Use it only for daily return time series
            if binary:
                if y_i > 0.:
                    y_i = [1., 0.]
                else:
                    y_i = [0., 1.]

                if scale: x_i = (np.array(x_i) - np.mean(x_i)) / np.std(x_i)
                
            else:
                timeseries = np.array(data[i:i+train+predict])
                shift_i = np.array(ewm[i:i+1])[0]
                if scale:
                    timeseries = timeseries/shift_i
                    y_i = timeseries[-1]
                    #y_i = (y_i - np.mean(timeseries[:-1])) / np.std(timeseries[:-1])
                    #x_i = (np.array(timeseries[:-1]) - np.mean(timeseries[:-1])) / np.std(timeseries[:-1])
                    x_i = timeseries[:-1]
                else:
                    x_i = timeseries[:-1]
                    y_i = timeseries[-1]
            
        except:
            break

        X.append(x_i)
        Y.append(y_i)
        shift.append(shift_i)

    return X, Y, shift


def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def create_Train_Test(data, percentage=0.8):
    Train = data[0:int(len(data) * percentage)]

    Test = data[int(len(data) * percentage):]

    return Train, Test


def create_Xt_Yt(X, y, percentage=0.8):
    X_train = X[0:int(len(X) * percentage)]
    Y_train = y[0:int(len(y) * percentage)]
    
    #X_train, Y_train = shuffle_in_unison(X_train, Y_train)

    X_test = X[int(len(X) * percentage):]
    Y_test = y[int(len(y) * percentage):]

    return X_train, X_test, Y_train, Y_test

def create_Xt_Yt_adaptive(X, y, shift, percentage=0.8):
    X_train = X[0:int(len(X) * percentage)]
    Y_train = y[0:int(len(y) * percentage)]
    shift_train = y[0:int(len(shift) * percentage)]
    
    #X_train, Y_train = shuffle_in_unison(X_train, Y_train)

    X_test = X[int(len(X) * percentage):]
    Y_test = y[int(len(y) * percentage):]
    shift_test = y[int(len(shift) * percentage):]

    return X_train, X_test, Y_train, Y_test, shift_train, shift_test


#from statsmodels.tsa.stattools import adfuller
#check if timeseries is stationary
# def test_stationarity(timeseries):
#
#     #Determing rolling statistics
#     rolmean = pd.rolling_mean(timeseries, window=12)
#     rolstd = pd.rolling_std(timeseries, window=12)
#
#     #Plot rolling statistics:
#     orig = plt.plot(timeseries, color='blue',label='Original')
#     mean = plt.plot(rolmean, color='red', label='Rolling Mean')
#     std = plt.plot(rolstd, color='black', label = 'Rolling Std')
#     plt.legend(loc='best')
#     plt.title('Rolling Mean & Standard Deviation')
#     plt.show(block=False)
#
#     #Perform Dickey-Fuller test:
#     print('Results of Dickey-Fuller Test:')
#     dftest = adfuller(timeseries.unstack(), autolag='AIC')
#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#     for key,value in dftest[4].items():
#         dfoutput['Critical Value (%s)'%key] = value
#     print(dfoutput)