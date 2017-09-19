from __future__ import print_function
import sys
import json
import numpy as np
import pandas
import math
import matplotlib.pylab as plt
#import talib

seed=7
np.random.seed(seed)  # for reproducibility

from processing import *


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from custom_callbacks import CriteriaStopping
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
#from hyperbolic_nonlinearities import AdaptativeAssymetricBiHyperbolic, AdaptativeBiHyperbolic, AdaptativeHyperbolicReLU, AdaptativeHyperbolic, PELU
#from keras.layers.advanced_activations import ParametricSoftplus, SReLU, PReLU, ELU, LeakyReLU, ThresholdedReLU


start_time = time.time()

# dataframe = pandas.read_csv('ibov_google_15jun2017_1min_15d.csv', sep = ',', usecols=[1],  engine='python', skiprows=8, decimal='.',header=None)
# dataset = dataframe[1].tolist()
dataframe = pandas.read_csv('minidolar/wdo.csv', sep = '|',  engine='python', decimal='.',header=0)
dataset = dataframe['fechamento'].tolist()

batch_size = 128
nb_epoch = 420
patience = 50
look_back = 7

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
        xt = np.array(xt)
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
        xt = np.array(xt)
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
        fp.write("-MINIDOLAR/MLP NN\n")

    hals = []

    TRAIN_SIZE = 30
    TARGET_TIME = 1
    LAG_SIZE = 1
    EMB_SIZE = 1
    
    X, Y = split_into_chunks(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=True)
    X, Y = np.array(X), np.array(Y)
    X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y, percentage=0.5)

    dados = X_train, X_test, Y_train, Y_test

    Xp, Yp = split_into_chunks(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
    Xp, Yp = np.array(Xp), np.array(Yp)
    X_trainp, X_testp, Y_trainp, Y_testp = create_Xt_Yt(Xp, Yp, percentage=0.5)

    dadosp = X_trainp, X_testp, Y_trainp, Y_testp

    testScore_aux = 999999
    f_aux = 0

    #for name in nonlinearities:
    for f in range(1,2):
        name='relu'
        model = Sequential()

        model.add(Dense(5, input_shape = (TRAIN_SIZE, )))
        model.add(Activation(name))

        for l in range(n_layers):
            model.add(Dense(5, input_shape = (TRAIN_SIZE, )))
            model.add(Activation(name))
        
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

    print("melhor parametro: %i" % f_aux)

if __name__ == "__main__":
   __main__(sys.argv[1:])
