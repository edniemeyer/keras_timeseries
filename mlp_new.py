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
from keras import regularizers
#from hyperbolic_nonlinearities import AdaptativeAssymetricBiHyperbolic, AdaptativeBiHyperbolic, AdaptativeHyperbolicReLU, AdaptativeHyperbolic, PELU
#from keras.layers.advanced_activations import ParametricSoftplus, SReLU, PReLU, ELU, LeakyReLU, ThresholdedReLU


start_time = time.time()


#USD-BRL
dataframe = pandas.read_csv('minidolar/wdo.csv', sep = '|',  engine='python', decimal='.',header=0)
dataset = dataframe['fechamento']
media  = dataframe['media'].tolist()

ewm_dolar = dataset.ewm(span=5, min_periods=5).mean()


#removendo NaN
dataset = dataset.iloc[4:]
ewm_dolar = ewm_dolar.iloc[4:]


batch_size = 128
nb_epoch = 5000
patience = 500
look_back = 7

TRAIN_SIZE = 30
TARGET_TIME = 1
LAG_SIZE = 1
EMB_SIZE = 1

X, Y = split_into_chunks(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False,
                                             scale=False)
X, Y = np.array(X), np.array(Y)
X_trainp, X_testp, Y_trainp, Y_testp = create_Xt_Yt(X, Y,  percentage=0.90)

def evaluate_model(model, name, n_layers, ep, normalization):
    #X_train, X_test, Y_train, Y_test = dataset
    #X_trainp, X_testp, Y_trainp, Y_testp = dadosp
    if (normalization == 'AN'):
        X_train, X_test, Y_train, Y_test, scaler, shift_train, shift_test = nn_an(dataset, ewm_dolar, TRAIN_SIZE,TARGET_TIME, LAG_SIZE)
    if (normalization == 'SW'):
        X_train, X_test, Y_train, Y_test, scaler_train, scaler_test = nn_sw(dataset,TRAIN_SIZE,TARGET_TIME, LAG_SIZE)
    if (normalization == 'MM'):
        X_train, X_test, Y_train, Y_test, scaler = nn_mm(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)
    if (normalization == 'ZS'):
        X_train, X_test, Y_train, Y_test, scaler = nn_zs(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)
    if (normalization == 'DS'):
        X_train, X_test, Y_train, Y_test, maximum = nn_ds(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)



    csv_logger = CSVLogger('output/%d_layers/%s.csv' % (n_layers, name))
    es = EarlyStopping(monitor='loss', patience=patience)
    #mcp = ModelCheckpoint('output/mnist_adaptative_%dx800/%s.checkpoint' % (n_layers, name), save_weights_only=True)
    #tb = TensorBoard(log_dir='output/mnist_adaptative_%dx800' % n_layers, histogram_freq=1, write_graph=False, write_images=False)

    
    #sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    # sgd = SGD(lr=0.64, momentum=0.8, nesterov=False)

    #optimizer = sgd
    optimizer = "adam"
    #optimizer = "adadelta"

    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    #history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=ep, verbose=0, validation_split=0.1, callbacks=[csv_logger,es])
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=ep, verbose=0, validation_split=0.1,
                        callbacks=[csv_logger])

    #trainScore = model.evaluate(X_train, Y_train, verbose=0)
    #print('Train Score: %f MSE (%f RMSE)' % (trainScore, math.sqrt(trainScore)))
    #testScore = model.evaluate(X_test, Y_test, verbose=0)
    #print('Test Score: %f MSE (%f RMSE)' % (testScore, math.sqrt(testScore)))

    # make predictions (scaled)
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)


    # invert predictions (back to original)
    if (normalization == 'AN'):
        X_trainp2, X_testp2, new_train_predicted, new_predicted = nn_an_den(X_train, X_test, trainPredict, testPredict, scaler, shift_train, shift_test)
    if (normalization == 'SW'):
        X_trainp2, X_testp2, new_train_predicted, new_predicted = nn_sw_den(X_train, X_test, trainPredict, testPredict, scaler_train, scaler_test)
    if (normalization == 'MM'):
        X_trainp2, X_testp2, new_train_predicted, new_predicted = nn_mm_den(X_train, X_test, trainPredict, testPredict, scaler)
    if (normalization == 'ZS'):
        X_trainp2, X_testp2, new_train_predicted, new_predicted = nn_zs_den(X_train, X_test, trainPredict, testPredict, scaler)
    if (normalization == 'DS'):
        X_trainp2, X_testp2, new_train_predicted, new_predicted = nn_ds_den(X_train, X_test, trainPredict, testPredict, maximum)

    # np.savetxt("output/previsto.csv", new_predicted)
    # np.savetxt("output/real.csv", Y_testp)
    # np.savetxt("output/previsto_treino.csv", new_train_predicted)
    # np.savetxt("output/real_treino.csv", Y_trainp)

    # np.savetxt("output/x_test-meu.csv", X_testp)
    # np.savetxt("output/y_test-meu.csv", Y_testp)
    # np.savetxt("output/x_treino-meu.csv", X_trainp)
    # np.savetxt("output/y_treino-meu.csv", Y_trainp)

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

    #normalizations = ['AN', 'SW', 'MM', 'ZS', 'DS']
    normalizations = ['AN', 'SW']
    with open("output/%d_layers/compare.csv" % n_layers, "a") as fp:
        fp.write("-MINIDOLAR/MLP NN\n")

    hals = []

    
    
    testScore_aux = 999999
    f_aux = 0

    #for name in nonlinearities:
    for normalization in normalizations:
    # for f in range(1,2):
        name='tanh'
        # normalization = 'AN'
        model = Sequential()

        model.add(Dense(12, input_shape = (TRAIN_SIZE, ), kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation(name))

        for l in range(n_layers):
            model.add(Dense(12, input_shape = (TRAIN_SIZE, )))
            model.add(Activation(name))
        
        model.add(Dropout(0.25))
        model.add(Dense(1))
        model.add(Activation('linear'))
        #model.summary()

        trainScore, testScore, epochs, optimizer = evaluate_model(model, name, n_layers,nb_epoch, normalization)
        # if(testScore_aux > testScore):
        #     testScore_aux=testScore
        #     f_aux = f

        elapsed_time = (time.time() - start_time)
        with open("output/%d_layers/compare.csv" % n_layers, "a") as fp:
            #fp.write("%i,%s,%f,%f,%d,%s --%s seconds\n" % (f, name, trainScore, testScore, epochs, optimizer, elapsed_time))
            fp.write("%s,%s,%f,%f,%d,%s --%s seconds\n" % (name, normalization, trainScore, testScore, epochs, optimizer, elapsed_time))

        model = None

    print("melhor parametro: %i" % f_aux)

if __name__ == "__main__":
   __main__(sys.argv[1:])
