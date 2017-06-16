from __future__ import print_function
import sys
import json
import numpy as np
import pandas
import math
#import talib

seed=7
np.random.seed(seed)  # for reproducibility

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import SGD
from keras.utils import np_utils
from custom_callbacks import CriteriaStopping
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from hyperbolic_nonlinearities import AdaptativeAssymetricBiHyperbolic, AdaptativeBiHyperbolic, AdaptativeHyperbolicReLU, AdaptativeHyperbolic, PELU
from keras.layers.advanced_activations import ParametricSoftplus, SReLU, PReLU, ELU, LeakyReLU, ThresholdedReLU


#dataframe = pandas.read_csv('DOLAR.csv', sep = ';', usecols=[1],  engine='python', skipfooter=3, decimal=',')
dataframe = pandas.read_csv('ibov_google_15jun2017_1min_15d.csv', sep = ',', usecols=[1],  engine='python', skiprows=8, decimal='.',header=None)
dataset = dataframe.values
dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = StandardScaler() #z-score
dataset = scaler.fit_transform(dataset) #não posso fazer scale no dataset inteiro.. apenas no treino

batch_size = 5
nb_epoch = 200
patience = 50
look_back = 7

def evaluate_model(model, dataset, name, n_layers, hals):
    X_train, Y_train, X_test, Y_test = dataset

    csv_logger = CSVLogger('output/%d_layers/%s.csv' % (n_layers, name))
    es = EarlyStopping(monitor='loss', patience=patience)
    #mcp = ModelCheckpoint('output/mnist_adaptative_%dx800/%s.checkpoint' % (n_layers, name), save_weights_only=True)
    #tb = TensorBoard(log_dir='output/mnist_adaptative_%dx800' % n_layers, histogram_freq=1, write_graph=False, write_images=False)

    
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

    optimizer = sgd
    #optimizer = "adam"
    #optimizer = "adadelta"

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    for i in range(nb_epoch):
	    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, verbose=0, callbacks=[csv_logger,es])
	    model.reset_states()

    #trainScore = model.evaluate(X_train, Y_train, verbose=0)
    #print('Train Score: %f MSE (%f RMSE)' % (trainScore, math.sqrt(trainScore)))
    #testScore = model.evaluate(X_test, Y_test, verbose=0)
    #print('Test Score: %f MSE (%f RMSE)' % (testScore, math.sqrt(testScore)))

    # make predictions
    trainPredict = model.predict(X_train,  batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(X_test,  batch_size=batch_size)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    Y_train = scaler.inverse_transform([Y_train])
    testPredict = scaler.inverse_transform(testPredict)
    Y_test = scaler.inverse_transform([Y_test])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(Y_train[0], trainPredict[:,0]))
    print('Train Score: %f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(Y_test[0], testPredict[:,0]))
    print('Test Score: %f RMSE' % (testScore))
    epochs = len(history.epoch)

    return trainScore, testScore, epochs, optimizer


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def load_dataset():
    train_size = 1003
    test_size = 1256 - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:train_size+test_size,:]
    print(len(train), len(test))
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    #trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    #testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # reshape input to be [samples, features, timesteps]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    
    print(trainX.shape[0], 'train samples')
    print(testX.shape[0], 'test samples')

    return trainX, trainY, testX, testY

def create_layer(name):
    if name == 'aabh':
        return AdaptativeAssymetricBiHyperbolic()
    elif name == 'abh':
        return AdaptativeBiHyperbolic()
    elif name == 'ah':
        return AdaptativeHyperbolic()
    elif name == 'ahrelu':
        return AdaptativeHyperbolicReLU()
    elif name == 'srelu':
        return SReLU()
    elif name == 'prelu':
        return PReLU()
    elif name == 'lrelu':
        return LeakyReLU()
    elif name == 'trelu':
        return ThresholdedReLU()
    elif name == 'elu':
        return ELU()
    elif name == 'pelu':
        return PELU()
    elif name == 'psoftplus':
        return ParametricSoftplus()
    elif name == 'sigmoid':
        return Activation('sigmoid')
    elif name == 'relu':
        return Activation('relu')
    elif name == 'tanh':
        return Activation('tanh')
    elif name == 'softplus':
        return Activation('softplus')

def __main__(argv):
    n_layers = int(argv[0])
    print(n_layers,'layers')

    dataset = load_dataset()
    
    #nonlinearities = ['aabh', 'abh', 'ah', 'ahrelu', 'srelu', 'prelu', 'lrelu', 'trelu', 'elu', 'pelu', 'psoftplus', 'sigmoid', 'relu', 'tanh', 'softplus']
    nonlinearities = ['aabh', 'abh', 'ah', 'sigmoid', 'relu', 'tanh']

    with open("output/%d_layers/compare.csv" % n_layers, "a") as fp:
        fp.write("-NN CONFIG: batch size %d, es patience %d, max_epoch %d, scaler %s, look_back %d\n" % (batch_size, patience, nb_epoch, scaler, look_back))
        fp.write("fn,RMSE_train,RMSE_test,epochs\n")

    hals = []

    for name in nonlinearities:
        model = Sequential()

        #model.add(Dense(4, input_dim=(look_back)))
        #model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
        HAL = create_layer(name)
        model.add(HAL)
        hals.append(HAL)
        for l in range(n_layers):
            model.add(Dense(4))
            HAL = create_layer(name)
            model.add(HAL)
            hals.append(HAL)
        model.add(Dense(1))
        model.add(HAL)
        model.summary()

        trainScore, testScore, epochs, optimizer = evaluate_model(model, dataset, name, n_layers, hals)

        with open("output/%d_layers/compare.csv" % n_layers, "a") as fp:
            fp.write("%s,%f,%f,%d,%s\n" % (name, trainScore, testScore, epochs, optimizer))

        model = None

if __name__ == "__main__":
   __main__(sys.argv[1:])
