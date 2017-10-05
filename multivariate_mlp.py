from __future__ import print_function
import sys
import json
import math
from utils import *

import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
import time


seed=7
np.random.seed(seed)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Conv1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint,ReduceLROnPlateau, TensorBoard
from hyperbolic_nonlinearities import *
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *

import seaborn as sns
start_time = time.time()
sns.despine()

batch_size = 128
nb_epoch = 420
patience = 50
look_back = 7
EMB_SIZE = 5 #numero de features

def evaluate_model(model, dataset, dadosp, name, n_layers, ep):
    X_train, X_test, Y_train, Y_test = dataset
    X_trainp, X_testp, Y_trainp, Y_testp = dadosp

    csv_logger = CSVLogger('output/%d_layers/%s.csv' % (n_layers, name))
    es = EarlyStopping(monitor='loss', patience=patience)
    #mcp = ModelCheckpoint('output/mnist_adaptative_%dx800/%s.checkpoint' % (n_layers, name), save_weights_only=True)
    #tb = TensorBoard(log_dir='output/mnist_adaptative_%dx800' % n_layers, histogram_freq=1, write_graph=False, write_images=False)

    
    #sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

    #optimizer = sgd
    optimizer = "adam"
    #optimizer = "adadelta"

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # reshape input to be [samples, time steps, features]
    #X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE))
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))
    #X_train = np.expand_dims(X_train, axis=2)
    #X_test = np.expand_dims(X_test, axis=2)

    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=ep, verbose=0, validation_split=0.1, callbacks=[csv_logger,es])

    #trainScore = model.evaluate(X_train, Y_train, verbose=0)
    #print('Train Score: %f MSE (%f RMSE)' % (trainScore, math.sqrt(trainScore)))
    #testScore = model.evaluate(X_test, Y_test, verbose=0)
    #print('Test Score: %f MSE (%f RMSE)' % (testScore, math.sqrt(testScore)))

    # make predictions (scaled)
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)
    
    
    # invert predictions (back to original)
    params = []
    for xt in X_testp:
        xt = np.array(xt[:,3]) # close
        mean_ = xt.mean()
        scale_ = xt.std()
        params.append([mean_, scale_])

    new_predicted = []

    for pred, par in zip(testPredict, params):
        a = pred*par[1]
        a += par[0]
        new_predicted.append(a)


    params2 = []
    for xt in X_trainp:
        xt = np.array(xt[:,3]) # close
        mean_ = xt.mean()
        scale_ = xt.std()
        params2.append([mean_, scale_])
        
    new_train_predicted= []

    for pred, par in zip(trainPredict, params2):
        a = pred*par[1]
        a += par[0]
        new_train_predicted.append(a)

    # calculate root mean squared error
    trainScore = mean_squared_error(new_train_predicted, Y_trainp)
    #print('Train Score: %f RMSE' % (trainScore))
    testScore = mean_squared_error(new_predicted, Y_testp)
    #print('Test Score: %f RMSE' % (testScore))
    epochs = len(history.epoch)

    # calculate root mean squared error
    trainScore = mean_squared_error(new_train_predicted, Y_trainp)
    #trainScore = mean_squared_error(trainPredict, Y_train)
    #print('Train Score: %f RMSE' % (trainScore))
    testScore = mean_squared_error(new_predicted, Y_testp)
    #testScore = mean_squared_error(testPredict, Y_test)
    #print('Test Score: %f RMSE' % (testScore))
    epochs = len(history.epoch)

    # fig = plt.figure()
    # plt.plot(Y_test[:150], color='black') # BLUE - trained RESULT
    # plt.plot(testPredict[:150], color='blue') # RED - trained PREDICTION
    #plt.plot(Y_testp[:150], color='green') # GREEN - actual RESULT
    #plt.plot(new_predicted[:150], color='red') # ORANGE - restored PREDICTION
    #plt.show()

    return trainScore, testScore, epochs, optimizer


def __main__(argv):
    n_layers = int(argv[0])
    print(n_layers,'layers')

    #nonlinearities = ['aabh', 'abh', 'ah', 'sigmoid', 'relu', 'tanh']
    nonlinearities = ['sigmoid', 'relu', 'tanh']
    #nonlinearities = ['relu']

    with open("output/%d_layers/compare.csv" % n_layers, "a") as fp:
        fp.write("-MINIDOLAR/MLP-Multi NN\n")

    hals = []
    #data_original = pd.read_csv('./data/AAPL1216.csv')[::-1]
    #data_original = pd.read_csv('ibov_google_15jun2017_1min_15d.csv', sep = ',',  engine='python', skiprows=8, decimal='.',header=None)

    # openp = data_original.ix[:, 4].tolist()
    # highp = data_original.ix[:, 2].tolist()
    # lowp = data_original.ix[:, 3].tolist()
    # closep = data_original.ix[:, 1].tolist()
    # volumep = data_original.ix[:, 5].tolist()

    data_original = pd.read_csv('minidolar/wdo.csv', sep = '|',  engine='python', decimal='.',header=0)

    averagep = data_original.ix[:, 1].tolist()
    openp = data_original.ix[:, 2].tolist()
    highp = data_original.ix[:, 3].tolist()
    lowp = data_original.ix[:, 4].tolist()
    closep = data_original.ix[:, 5].tolist()
    volumep = data_original.ix[:, 6].tolist()

    # data_chng = data_original.ix[:, 'Adj Close'].pct_change().dropna().tolist()

    WINDOW = 30
    TRAIN_SIZE=WINDOW
    
    STEP = 1
    FORECAST = 1

    X, Y = [], []
    for i in range(0, len(data_original), STEP): 
        try:
            #a = averagep[i:i+WINDOW+FORECAST]
            o = openp[i:i+WINDOW+FORECAST]
            h = highp[i:i+WINDOW+FORECAST]
            l = lowp[i:i+WINDOW+FORECAST]
            c = closep[i:i+WINDOW+FORECAST]
            v = volumep[i:i+WINDOW+FORECAST]

            #a = (np.array(a) - np.mean(a)) / np.std(a)
            o = (np.array(o) - np.mean(o)) / np.std(o)
            h = (np.array(h) - np.mean(h)) / np.std(h)
            l = (np.array(l) - np.mean(l)) / np.std(l)
            c = (np.array(c) - np.mean(c)) / np.std(c)
            v = (np.array(v) - np.mean(v)) / np.std(v)

            x_i = closep[i:i+WINDOW]
            y_i = closep[i+WINDOW+FORECAST]  

            timeseries = np.array(c)
            #x_i = np.column_stack((a[:-1], o[:-1], h[:-1], l[:-1], c[:-1], v[:-1]))
            x_i = np.column_stack((o[:-1], h[:-1], l[:-1], c[:-1], v[:-1]))
            y_i = timeseries[-1]

        except Exception as e:
            break

        X.append(x_i)
        Y.append(y_i)

    X, Y = np.array(X), np.array(Y)
    X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y, 0.5)
    dados = X_train, X_test, Y_train, Y_test
    
    Xp, Yp = [], []
    for i in range(0, len(data_original), STEP): 
        try:
            #a = averagep[i:i+WINDOW+FORECAST]
            o = openp[i:i+WINDOW]
            h = highp[i:i+WINDOW]
            l = lowp[i:i+WINDOW]
            c = closep[i:i+WINDOW]
            v = volumep[i:i+WINDOW]
            forecasted_c = closep[i+WINDOW+FORECAST]
            #scaling FORECASTED close
            forecasted_c = (forecasted_c - np.mean(c)) / np.std(c)

            #scaling WINDOW data
            #a = (np.array(a) - np.mean(a)) / np.std(a)
            o = (np.array(o) - np.mean(o)) / np.std(o)
            h = (np.array(h) - np.mean(h)) / np.std(h)
            l = (np.array(l) - np.mean(l)) / np.std(l)
            c = (np.array(c) - np.mean(c)) / np.std(c)
            v = (np.array(v) - np.mean(v)) / np.std(v)

            #timeseries = np.array(c.append(forecasted_c))
            #x_i = np.column_stack((a[:-1], o[:-1], h[:-1], l[:-1], c[:-1], v[:-1]))
            x_i = np.column_stack((o, h, l, c, v))
            y_i = forecasted_c

        except Exception as e:
            break

        Xp.append(x_i)
        Yp.append(y_i)

    
    Xp, Yp = np.array(Xp), np.array(Yp)
    X_trainp, X_testp, Y_trainp, Y_testp = create_Xt_Yt(Xp, Yp, percentage=0.5)
    dadosp = X_trainp, X_testp, Y_trainp, Y_testp


    for f in range(1,2):
        name='relu'
        model = Sequential()

        model.add(Dense(5, input_shape = (TRAIN_SIZE, EMB_SIZE)))
        model.add(Activation(name))

        for l in range(n_layers):
            model.add(Dense(5, input_shape = (TRAIN_SIZE, EMB_SIZE)))
            model.add(Activation(name))
        
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('linear'))
        #model.summary()

        trainScore, testScore, epochs, optimizer = evaluate_model(model, dados, dadosp, name, n_layers,nb_epoch)
        # if(testScore_aux > testScore):
        #     testScore_aux=testScore
        #     f_aux = f

        elapsed_time = (time.time() - start_time)
        with open("output/%d_layers/compare.csv" % n_layers, "a") as fp:
            #fp.write("%i,%s,%f,%f,%d,%s --%s seconds\n" % (f, name, trainScore, testScore, epochs, optimizer, elapsed_time))
            fp.write("%s,%f,%f,%d,%s --%s seconds\n" % (name, trainScore, testScore, epochs, optimizer, elapsed_time))

        model = None

        #print("melhor parametro: %i" % f_aux)

if __name__ == "__main__":
   __main__(sys.argv[1:])

