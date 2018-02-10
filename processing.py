import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, classification_report
import matplotlib.pylab as plt
import datetime as dt
import time
from normalizer import *

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
    for i in range(0, len(data)-train-predict, step):
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
    for i in range(0, len(data)-train-predict, step):
        try:
            # Use it only for daily return time series
            if binary:
                x_i = data[i:i + train]
                y_i = data[i + train + predict]
                if y_i > 0.:
                    y_i = [1., 0.]
                else:
                    y_i = [0., 1.]

                if scale: x_i = (np.array(x_i) - np.mean(x_i)) / np.std(x_i)
                
            else:
                timeseries = np.array(data[i:i+train+predict])
                shift_i = np.array(ewm[i:i+1])[0]
                #shift_i = np.mean(timeseries[:-1])

                if scale:
                    timeseries = timeseries-shift_i
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


def split_into_chunks_adaptive_try(data, ewm, train, predict, step):
    X, Y, shift, R = [], [], [], []
    data, ewm = np.array(data), np.array(ewm)
    #generating new sequence R with adaptive normalization

    for i in range(0, len(data)-train-predict, step):
        try:
            for j in range(1, len(data[i:i+train+predict]) + 1, 1):  #for not having 0 division, it has to start at 1, so add that 1 in the end
                R.append(data[i:i+train+predict][int(np.ceil(j / (train + predict)) * (j - 1) % (train + predict))]
                         / ewm[i:i+train+predict][int(np.ceil(j / (train + predict)))])
                shift.append(ewm[i:i+train+predict][int(np.ceil(j / (train + predict)))])

            timeseries = np.array(R[i:i + train + predict])
            x_i = timeseries[:-1]
            y_i = timeseries[-1]

        except:
            break

        X.append(x_i)
        Y.append(y_i)

    return X, Y, shift, R



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
    shift_train = shift[0:int(len(shift) * percentage)]
    
    #X_train, Y_train = shuffle_in_unison(X_train, Y_train)

    X_test = X[int(len(X) * percentage):]
    Y_test = y[int(len(y) * percentage):]
    shift_test = shift[int(len(shift) * percentage):]

    return X_train, X_test, Y_train, Y_test, shift_train, shift_test


from statsmodels.tsa.stattools import adfuller
#check if timeseries is stationary
def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12,center=False).mean()
    rolstd = timeseries.rolling(window=12,center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries.values, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)



#minmax normalization without sliding windows

def nn_mm(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE):
    # X, Y = split_into_chunks(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
    # X, Y = np.array(X), np.array(Y)
    # X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y, percentage=0.35)
    #
    # X_normalizado, scaler = minMaxNormalize(X_train.tolist())

    train, test = create_Train_Test(dataset, 0.80)
    train_normalizado, scaler = minMaxNormalize(train.values.reshape(-1,1))

    dataset_norm = minMaxNormalizeOver(dataset.values.reshape(-1,1), scaler)
    #dataset_norm, scaler = minMaxNormalize(dataset.values.reshape(-1, 1))

    X, Y = split_into_chunks(dataset_norm.reshape(-1), TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
    X, Y = np.array(X), np.array(Y)
    X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y, percentage=0.80)
    return X_train, X_test, Y_train, Y_test, scaler

def nn_mm_den(X_train, X_test, Y_train, Y_test, scaler):
    X_train = minMaxDenormalize(X_train, scaler)
    X_test = minMaxDenormalize(X_test, scaler)
    Y_train = minMaxDenormalize(Y_train, scaler)
    Y_test = minMaxDenormalize(Y_test, scaler)
    X_train, X_test, Y_train, Y_test = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)
    return X_train, X_test, Y_train, Y_test


#minmax normalization with sliding windows

def nn_sw(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE):

    X, Y = split_into_chunks(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
    X, Y = np.array(X), np.array(Y)
    X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y, percentage=0.80)

    X_train_n, X_test_n, Y_train_n, Y_test_n, scaler_train, scaler_test = [],[],[],[],[],[]
    for i in range(X_train.shape[0]):
        X_normalizado, scaler = minMaxNormalize(X_train[i].reshape(-1,1)) # shape(30,1)
        X_train_n.append(X_normalizado.reshape(-1)) # shape(30)
        Y_train_n.append(minMaxNormalizeOver(Y_train[i], scaler).reshape(-1))
        scaler_train.append(scaler)

    for i in range(X_test.shape[0]):
        X_normalizado, scaler = minMaxNormalize(X_test[i].reshape(-1, 1))  # shape(30,1)
        X_test_n.append(X_normalizado.reshape(-1))  # shape(30)
        Y_test_n.append(minMaxNormalizeOver(Y_test[i], scaler).reshape(-1))
        scaler_test.append(scaler)

    X_train, X_test, Y_train, Y_test = np.array(X_train_n), np.array(X_test_n), np.array(Y_train_n), np.array(Y_test_n)
    return X_train, X_test, Y_train, Y_test, scaler_train, scaler_test

def nn_sw_den(X_train, X_test, Y_train, Y_test, scaler_train, scaler_test):
    X_train_d, X_test_d, Y_train_d, Y_test_d = [], [], [], []
    for i in range(X_train.shape[0]):
        X_denormalizado = minMaxDenormalize(X_train[i].reshape(-1, 1), scaler_train[i]).reshape(-1)
        X_train_d.append(X_denormalizado)
        Y_denormalizado = minMaxDenormalize(Y_train[i].reshape(-1, 1), scaler_train[i]).reshape(-1)
        Y_train_d.append(Y_denormalizado)

    for i in range(X_test.shape[0]):
        X_denormalizado = minMaxDenormalize(X_test[i].reshape(-1, 1), scaler_test[i]).reshape(-1)
        X_test_d.append(X_denormalizado)
        Y_denormalizado = minMaxDenormalize(Y_test[i].reshape(-1, 1), scaler_test[i]).reshape(-1)
        Y_test_d.append(Y_denormalizado)

    X_train, X_test, Y_train, Y_test = np.array(X_train_d), np.array(X_test_d), np.array(Y_train_d), np.array(Y_test_d)
    return X_train, X_test, Y_train, Y_test


#z-score normalization
def nn_zs(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE):
    train, test = create_Train_Test(dataset, 0.80)
    train_normalizado, scaler = zNormalize(train.values.reshape(-1,1))

    dataset_norm = zNormalizeOver(dataset.values.reshape(-1,1), scaler)

    X, Y = split_into_chunks(dataset_norm.reshape(-1), TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
    X, Y = np.array(X), np.array(Y)
    X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y, percentage=0.80)
    return X_train, X_test, Y_train, Y_test, scaler

def nn_zs_den(X_train, X_test, Y_train, Y_test, scaler):
    X_train = zDenormalize(X_train, scaler)
    X_test = zDenormalize(X_test, scaler)
    Y_train = zDenormalize(Y_train, scaler)
    Y_test = zDenormalize(Y_test, scaler)
    X_train, X_test, Y_train, Y_test = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)
    return X_train, X_test, Y_train, Y_test

#decimal normalization
def nn_ds(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE):
    train, test = create_Train_Test(dataset, 0.80)
    train_normalizado = decimalNormalize(train.values.reshape(-1,1))

    dataset_norm = decimalNormalizeOver(dataset.values.reshape(-1,1), max(train))

    X, Y = split_into_chunks(dataset_norm.reshape(-1), TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
    X, Y = np.array(X), np.array(Y)
    X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y, percentage=0.80)
    return X_train, X_test, Y_train, Y_test, max(train)

def nn_ds_den(X_train, X_test, Y_train, Y_test, maximum):
    X_train = decimalDenormalize(X_train, maximum)
    X_test = decimalDenormalize(X_test, maximum)
    Y_train = decimalDenormalize(Y_train, maximum)
    Y_test = decimalDenormalize(Y_test, maximum)
    X_train, X_test, Y_train, Y_test = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)
    return X_train, X_test, Y_train, Y_test


#adaptive normalization (Adaptive Normalization: A Novel Data Normalization Approach for  Non-Stationary Time Series)

def nn_an(dataset, ewm, TRAIN_SIZE, TARGET_TIME, LAG_SIZE):
    X, Y, shift = split_into_chunks_adaptive(dataset, ewm, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False,
                                             scale=True)
    X, Y, shift = np.array(X), np.array(Y), np.array(shift)
    X_train, X_test, Y_train, Y_test, shift_train, shift_test = create_Xt_Yt_adaptive(X, Y, shift, percentage=0.80)
    sample_normalizado, scaler = minMaxNormalize(X_train.reshape(-1,1))# global scaler over sample set, as said on the article
    X_train_n, X_test_n, Y_train_n, Y_test_n= [], [], [], []
    for i in range(X_train.shape[0]):
        X_normalizado = minMaxNormalizeOver(X_train[i].reshape(-1, 1), scaler)  # shape(30,1)
        X_train_n.append(X_normalizado.reshape(-1))  # shape(30)
        Y_train_n.append(minMaxNormalizeOver(Y_train[i], scaler).reshape(-1))

    for i in range(X_test.shape[0]):
        X_normalizado = minMaxNormalizeOver(X_test[i].reshape(-1, 1), scaler)  # shape(30,1)
        X_test_n.append(X_normalizado.reshape(-1))  # shape(30)
        Y_test_n.append(minMaxNormalizeOver(Y_test[i], scaler).reshape(-1))

    X_train, X_test, Y_train, Y_test = np.array(X_train_n), np.array(X_test_n), np.array(Y_train_n), np.array(Y_test_n)
    return X_train, X_test, Y_train, Y_test, scaler, shift_train, shift_test


def nn_an_den(X_train, X_test, Y_train, Y_test, scaler, shift_train, shift_test):
    X_train_d, X_test_d, Y_train_d, Y_test_d = [], [], [], []
    for i in range(X_train.shape[0]):
        X_denormalizado = minMaxDenormalize(X_train[i].reshape(-1, 1), scaler).reshape(-1)
        X_denormalizado = X_denormalizado + shift_train[i]
        X_train_d.append(X_denormalizado)
        Y_denormalizado = minMaxDenormalize(Y_train[i].reshape(-1, 1), scaler).reshape(-1)
        Y_denormalizado = Y_denormalizado + shift_train[i]
        Y_train_d.append(Y_denormalizado)

    for i in range(X_test.shape[0]):
        X_denormalizado = minMaxDenormalize(X_test[i].reshape(-1, 1), scaler).reshape(-1)
        X_denormalizado = X_denormalizado + shift_test[i]
        X_test_d.append(X_denormalizado)
        Y_denormalizado = minMaxDenormalize(Y_test[i].reshape(-1, 1), scaler).reshape(-1)
        Y_denormalizado = Y_denormalizado + shift_test[i]
        Y_test_d.append(Y_denormalizado)

    X_train, X_test, Y_train, Y_test = np.array(X_train_d), np.array(X_test_d), np.array(Y_train_d), np.array(Y_test_d)
    return X_train, X_test, Y_train, Y_test