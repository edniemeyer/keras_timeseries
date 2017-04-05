from __future__ import print_function
import sys
import json
import numpy as np
import pandas
import math
seed=7
np.random.seed(seed)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from custom_callbacks import CriteriaStopping
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from hyperbolic_nonlinearities import AdaptativeAssymetricBiHyperbolic, AdaptativeBiHyperbolic, AdaptativeHyperbolicReLU, AdaptativeHyperbolic, PELU
from keras.layers.advanced_activations import ParametricSoftplus, SReLU, PReLU, ELU, LeakyReLU, ThresholdedReLU


dataframe = pandas.read_csv('DOLAR.csv', sep = ';', usecols=[1],  engine='python', skipfooter=3, decimal=',')
dataset = dataframe.values
dataset = dataset.astype('float32')



batch_size = 20
nb_epoch = 1000
patience = 50
look_back = 6

def evaluate_model(model, dataset, name, n_layers, hals):
    X_train, Y_train, X_test, Y_test = dataset

    csv_logger = CSVLogger('output/%d_layers/%s.csv' % (n_layers, name))
    es = EarlyStopping(monitor='loss', patience=patience)
    #mcp = ModelCheckpoint('output/mnist_adaptative_%dx800/%s.checkpoint' % (n_layers, name), save_weights_only=True)
    #tb = TensorBoard(log_dir='output/mnist_adaptative_%dx800' % n_layers, histogram_freq=1, write_graph=False, write_images=False)

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    #model.compile(loss='mean_squared_error', optimizer='adam')
    #model.compile(loss='mean_squared_error', optimizer='adadelta')

    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, callbacks=[csv_logger])

    trainScore = model.evaluate(X_train, Y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(X_test, Y_test, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

    epochs = len(history.epoch)

    return trainScore, testScore, epochs


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
    
    nonlinearities = ['aabh', 'abh', 'ah', 'ahrelu', 'srelu', 'prelu', 'lrelu', 'trelu', 'elu', 'pelu', 'psoftplus', 'sigmoid', 'relu', 'tanh', 'softplus']

    with open("output/%d_layers/compare.csv" % n_layers, "a") as fp:
        fp.write("-NN CONFIG: batch size %d, es patience %d, max_epoch %d\n" % (batch_size, patience, nb_epoch))
        fp.write("fn,RMSE_train,RMSE_test,epochs\n")

    hals = []

    for name in nonlinearities:
        model = Sequential()

        model.add(Dense(4, input_dim=(look_back)))
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

        trainScore, testScore, epochs = evaluate_model(model, dataset, name, n_layers, hals)

        with open("output/%d_layers/compare.csv" % n_layers, "a") as fp:
            fp.write("%s,%f,%f,%d\n" % (name, trainScore, testScore, epochs))

        model = None

if __name__ == "__main__":
   __main__(sys.argv[1:])
