from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED'] = '0'
import sys
import json
import numpy as np
import pandas
import math
import tensorflow as tf
import random
import matplotlib.pylab as plt
#import talib

seed = [4395,3129,277,9871,5183,6082,810,6979,2654,5765]

def set_seeds(seed):
    np.random.seed(seed)  # for reproducibility
    tf.set_random_seed(seed)
    random.seed(seed)


from processing import *


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from custom_callbacks import CriteriaStopping
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from hyperbolic_nonlinearities import *
from keras import regularizers
#from hyperbolic_nonlinearities import AdaptativeAssymetricBiHyperbolic, AdaptativeBiHyperbolic, AdaptativeHyperbolicReLU, AdaptativeHyperbolic, PELU
#from keras.layers.advanced_activations import ParametricSoftplus, SReLU, PReLU, ELU, LeakyReLU, ThresholdedReLU

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)


from keras import backend as K

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


start_time = time.time()

#USD-BRL
dataframe = pandas.read_csv('minidolar/wdo.csv', sep = '|',  engine='python', decimal='.',header=0)
dataset_original = dataframe['fechamento']


batch_size = 64
nb_epoch = 100
patience = 1000

TRAIN_SIZE = 30
TARGET_TIME = 1
LAG_SIZE = 1
EMB_SIZE = 1


def evaluate_model(model, name, n_layers, ep, normalization, TRAIN_SIZE, dataset, ewm_dolar, type):
    if (normalization == 'AN'):
        X_train, X_test, Y_train, Y_test, scaler, shift_train, shift_test, X_trainp, X_testp, Y_trainp, Y_testp = nn_an_type(dataset, ewm_dolar, TRAIN_SIZE,TARGET_TIME, LAG_SIZE, type)
    if (normalization == 'SW'):
        X_train, X_test, Y_train, Y_test, scaler_train, scaler_test, X_trainp, X_testp, Y_trainp, Y_testp = nn_sw(dataset,TRAIN_SIZE,TARGET_TIME, LAG_SIZE)
    if (normalization == 'MM'):
        X_train, X_test, Y_train, Y_test, scaler, X_trainp, X_testp, Y_trainp, Y_testp = nn_mm(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)
    if (normalization == 'ZS'):
        X_train, X_test, Y_train, Y_test, scaler, X_trainp, X_testp, Y_trainp, Y_testp = nn_zs(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)
    if (normalization == 'DS'):
        X_train, X_test, Y_train, Y_test, maximum, X_trainp, X_testp, Y_trainp, Y_testp = nn_ds(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)


    csv_logger = CSVLogger('output/%d_layers/%s_%s.csv' % (n_layers, name, normalization))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss')
    es = EarlyStopping(monitor='val_loss', patience=patience)
    #mcp = ModelCheckpoint('output/mnist_adaptative_%dx800/%s.checkpoint' % (n_layers, name), save_weights_only=True)
    #tb = TensorBoard(log_dir='output/mnist_adaptative_%dx800' % n_layers, histogram_freq=1, write_graph=False, write_images=False)

    
    #sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

    #optimizer = sgd
    optimizer = "adam"
    #optimizer = "adadelta"

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
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
    if (normalization == 'AN'):
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_an_den_type(X_train, X_test, trainPredict, testPredict, scaler, shift_train, shift_test, type)

    if (normalization == 'SW'):
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_sw_den(X_train, X_test, trainPredict, testPredict, scaler_train, scaler_test)

    if (normalization == 'MM'):
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_mm_den(X_train, X_test, trainPredict, testPredict, scaler)

    if (normalization == 'ZS'):
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_zs_den(X_train, X_test, trainPredict, testPredict, scaler)

    if (normalization == 'DS'):
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_ds_den(X_train, X_test, trainPredict, testPredict, maximum)

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(new_train_predicted, Y_trainp))
    #print('Train Score: %f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(new_predicted, Y_testp))
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

    # normalizations = ['AN', 'SW', 'MM', 'ZS', 'DS']
    #normalizations = ['DS']
    normalizations = ['AN']
    type = 'c'
    with open("output/%d_layers/compare.csv" % n_layers, "a") as fp:
        fp.write("-MINIDOLAR/CONV NN %s\n" % type)

    hals = []

    
    testScore_aux = 999999
    f_aux = 0

    for o in range(2, 30, 5):
        for p in seed:
            TRAIN_SIZE = o

            set_seeds(p)

            k = 3

            ewm_dolar = dataset_original.ewm(span=k, min_periods=k).mean()

            # removendo NaN
            dataset = np.array(dataset_original.iloc[k - 1:])
            ewm_dolar = np.array(ewm_dolar.iloc[k - 1:])

	    #for name in nonlinearities:
            for normalization in normalizations:
            # for f in range(1,2):
                name='tanh'
                model = Sequential()

		#model.add(Dense(500, input_shape = (TRAIN_SIZE, )))
	        #model.add(Activation(name))
                model.add(Conv1D(input_shape = (TRAIN_SIZE, EMB_SIZE),filters=(o-1),kernel_size=2,activation=name,padding='causal',strides=1,
                    kernel_regularizer=regularizers.l2(0.01)))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Dropout(0.25))

                for l in range(n_layers):
                    model.add(Conv1D(input_shape = (TRAIN_SIZE, EMB_SIZE),filters=(o-1),kernel_size=2,activation=name,padding='causal',strides=1,
                    kernel_regularizer=regularizers.l2(0.01)))
                    model.add(MaxPooling1D(pool_size=1))
                    model.add(Dropout(0.25))

		    #model.add(Dense(5))
        	    #model.add(Dropout(0.25))
		    #model.add(Activation(name))
                model.add(Flatten())
                model.add(Dense(1))
                model.add(Activation(name))
                #model.summary()

                trainScore, testScore, epochs, optimizer = evaluate_model(model, name, n_layers,nb_epoch, normalization, TRAIN_SIZE, dataset, ewm_dolar, type)
                elapsed_time = (time.time() - start_time)

                with open("output/%d_layers/compare.csv" % n_layers, "a") as fp:
	             #fp.write("%i,%s,%f,%f,%d,%s --%s seconds\n" % (f, name, trainScore, testScore, epochs, optimizer, elapsed_time))
                     fp.write("w=%i,k=%i,%s,%s,%f,%f,%d,%s --%s seconds\n" % (o,p, name, normalization, trainScore, testScore, epochs, optimizer, elapsed_time))
                model = None

    #print("melhor parametro: %i" % f_aux)

if __name__ == "__main__":
   __main__(sys.argv[1:])
