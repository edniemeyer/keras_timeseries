import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np
import pandas
import math
from tensorflow import set_random_seed
import matplotlib.pylab as plt

# import talib

seed = 7
np.random.seed(seed)  # for reproducibility
set_random_seed(seed)

from processing import *

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import regularizers

start_time = time.time()

# USD-BRL
dataframe = pandas.read_csv('furnas-vazoes-medias-mensais-m3s.csv', sep=',', engine='python', header=0)
dataset_original = dataframe['furnas']

batch_size = 64
nb_epoch = 100
patience = 1000
TARGET_TIME = 1
LAG_SIZE = 1
EMB_SIZE = 1


def evaluate_model(model, name, n_layers, ep, normalization, TRAIN_SIZE, dataset, ewm_dolar, type):
    # X_train, X_test, Y_train, Y_test = dataset
    # X_trainp, X_testp, Y_trainp, Y_testp = dadosp
    if (normalization == 'AN'):
        X_train, X_test, Y_train, Y_test, scaler, shift_train, shift_test, X_trainp, X_testp, Y_trainp, Y_testp = nn_an_type(
            dataset, ewm_dolar, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, type)
    if (normalization == 'SW'):
        X_train, X_test, Y_train, Y_test, scaler_train, scaler_test, X_trainp, X_testp, Y_trainp, Y_testp = nn_sw(
            dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)
    if (normalization == 'MM'):
        X_train, X_test, Y_train, Y_test, scaler, X_trainp, X_testp, Y_trainp, Y_testp  = nn_mm(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)
    if (normalization == 'ZS'):
        X_train, X_test, Y_train, Y_test, scaler, X_trainp, X_testp, Y_trainp, Y_testp  = nn_zs(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)
    if (normalization == 'DS'):
        X_train, X_test, Y_train, Y_test, maximum, X_trainp, X_testp, Y_trainp, Y_testp  = nn_ds(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)

    csv_logger = CSVLogger('output/%d_layers/%s_%s.csv' % (n_layers, name, normalization))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss')
    es = EarlyStopping(monitor='val_loss', patience=patience)
    # mcp = ModelCheckpoint('output/mnist_adaptative_%dx800/%s.checkpoint' % (n_layers, name), save_weights_only=True)
    # tb = TensorBoard(log_dir='output/mnist_adaptative_%dx800' % n_layers, histogram_freq=1, write_graph=False, write_images=False)

    # sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    # sgd = SGD(lr=0.64, momentum=0.8, nesterov=False)

    # optimizer = sgd
    # optimizer = Adam(lr=0.002)
    optimizer = 'adam'
    # optimizer = "adadelta"
    # optimizer = "nadam"

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=ep, verbose=0, validation_split=0.1,
                        callbacks=[csv_logger, es])

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    # trainScore = model.evaluate(X_train, Y_train, verbose=0)
    # print('Train Score: %f MSE (%f RMSE)' % (trainScore, math.sqrt(trainScore)))
    # testScore = model.evaluate(X_test, Y_test, verbose=0)
    # print('Test Score: %f MSE (%f RMSE)' % (testScore, math.sqrt(testScore)))

    # make predictions (scaled)
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)

    # invert predictions (back to original)
    if (normalization == 'AN'):
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_an_den_type(X_train, X_test, trainPredict, testPredict,
                                                                            scaler, shift_train, shift_test, type)

    if (normalization == 'SW'):
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_sw_den(X_train, X_test, trainPredict, testPredict,
                                                                            scaler_train, scaler_test)

    if (normalization == 'MM'):
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_mm_den(X_train, X_test, trainPredict, testPredict,
                                                                            scaler)

    if (normalization == 'ZS'):
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_zs_den(X_train, X_test, trainPredict, testPredict,
                                                                            scaler)

    if (normalization == 'DS'):
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_ds_den(X_train, X_test, trainPredict, testPredict,
                                                                            maximum)

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
    # print('Train Score: %f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(new_predicted, Y_testp))
    # print('Test Score: %f RMSE' % (testScore))
    epochs = len(history.epoch)

    # fig = plt.figure()
    # plt.plot(Y_test[:150], color='black') # BLUE - trained RESULT
    # plt.plot(testPredict[:150], color='blue') # RED - trained PREDICTION
    # plt.plot(Y_testp[:150], color='green') # GREEN - actual RESULT
    # plt.plot(new_predicted[:150], color='red') # ORANGE - restored PREDICTION
    # plt.show()

    return trainScore, testScore, epochs, optimizer


def __main__(argv):
    n_layers = int(argv[0])
    print(n_layers, 'layers')

    # nonlinearities = ['aabh', 'abh', 'ah', 'sigmoid', 'relu', 'tanh']
    nonlinearities = ['sigmoid', 'relu', 'tanh']
    # nonlinearities = ['relu']

    #normalizations = ['DS']
    normalizations = ['AN']
    type = 'c'
    with open("output/%d_layers/compare.csv" % n_layers, "a") as fp:
        fp.write("-FURNAS/MLP NN %s\n" % type)

    hals = []

    # best parameters without outlier removal: TRAIN_SIZE= 7 k=25
    # with outlier removal: TRAIN_SIZE=4 k=3

    for o in range(3,15, 3):
        for p in range(0, 10):
            TRAIN_SIZE = 7

            k = 8

            ewm_dolar = dataset_original.ewm(span=k, min_periods=k).mean()

            # removendo NaN
            dataset = np.array(dataset_original.iloc[k - 1:])
            ewm_dolar = np.array(ewm_dolar.iloc[k - 1:])

            # for name in nonlinearities:
            for normalization in normalizations:
                # for f in range(1,2):
                name = 'tanh'
                model = Sequential()

                model.add(Dense(o, input_shape=(TRAIN_SIZE,), kernel_initializer='glorot_uniform',
                                kernel_regularizer=regularizers.l2(0.01)))
                model.add(Activation(name))
                model.add(Dropout(0.25))

                for l in range(n_layers):
                    model.add(Dense(o, input_shape=(TRAIN_SIZE,), kernel_initializer='glorot_uniform',
                                kernel_regularizer=regularizers.l2(0.01)))
                    model.add(Activation(name))
                    model.add(Dropout(0.25))
                # model.add(Dropout(0.25))

                model.add(Dense(1))
                model.add(Activation(name))
                # model.summary()

                trainScore, testScore, epochs, optimizer = evaluate_model(model, name, n_layers, nb_epoch,
                                                                          normalization, TRAIN_SIZE, dataset, ewm_dolar, type=type)
                # if(testScore_aux > testScore):
                #     testScore_aux=testScore
                #     f_aux = f

                elapsed_time = (time.time() - start_time)
                with open("output/%d_layers/compare.csv" % n_layers, "a") as fp:
                    # fp.write("%i,%s,%f,%f,%d,%s --%s seconds\n" % (f, name, trainScore, testScore, epochs, optimizer, elapsed_time))
                    fp.write("w=%i,k=%i,%s,%s,%f,%f,%d,%s --%s seconds\n" % (
                    o, p, name, normalization, trainScore, testScore, epochs, optimizer, elapsed_time))
                model = None


if __name__ == "__main__":
    __main__(sys.argv[1:])
