from __future__ import print_function
import sys
import math
from processing import *

import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
import time


seed=7
np.random.seed(seed)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint,ReduceLROnPlateau, TensorBoard
from hyperbolic_nonlinearities import *
from keras import regularizers
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *

start_time = time.time()

batch_size = 64
nb_epoch = 1000
patience = 500
EMB_SIZE = 4 #numero de features


TRAIN_SIZE = 30
TARGET_TIME = 1
LAG_SIZE = 1


data_original = pd.read_csv('btc-usd.csv', sep = ',',  engine='python', decimal='.',header=0)

closep = data_original['close']
ewm = closep.ewm(span=30, min_periods=30).mean()


#removendo NaN
data_original = data_original.iloc[29:]
ewm = ewm.iloc[29:]

#averagep = data_original.ix[:, 1].tolist()
openp = data_original['open'].tolist()
highp = data_original['high'].tolist()
lowp = data_original['low'].tolist()
closep = data_original['close'].tolist()
#volumep = data_original.ix[:, 6].tolist()


dataset = np.column_stack((openp, highp, lowp, closep))


X, Y = split_into_chunks(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
X, Y = np.array(X), np.array(Y)

X_trainp, X_testp, Y_trainp, Y_testp = create_Xt_Yt(X, Y)
Y_trainp, Y_testp = Y_trainp[:,3], Y_testp[:,3] #getting just close as target

def evaluate_model(model, name, n_layers, ep, normalization):
    if (normalization == 'AN'):
        X_train, X_test, Y_train, Y_test, scaler, shift_train, shift_test = nn_an(dataset, ewm, TRAIN_SIZE,TARGET_TIME, LAG_SIZE)
    if (normalization == 'SW'):
        X_train, X_test, Y_train, Y_test, scaler_train, scaler_test = nn_sw(dataset,TRAIN_SIZE,TARGET_TIME, LAG_SIZE)
    if (normalization == 'MM'):
        X_train, X_test, Y_train, Y_test, scaler = nn_mm(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)
    if (normalization == 'ZS'):
        X_train, X_test, Y_train, Y_test, scaler = nn_zs(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)
    if (normalization == 'DS'):
        X_train, X_test, Y_train, Y_test, maximum = nn_ds(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)



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
    # X_train = np.reshape(X_train, (X_train.shape[0], int(X_train.shape[1]/EMB_SIZE), EMB_SIZE))
    # X_test = np.reshape(X_test, (X_test.shape[0], int(X_test.shape[1]/EMB_SIZE), EMB_SIZE))
    #X_train = np.expand_dims(X_train, axis=2)
    #X_test = np.expand_dims(X_test, axis=2)

    Y_train = Y_train[:,3]
    Y_test = Y_test[:,3]

    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=ep, verbose=0, validation_split=0.1, callbacks=[csv_logger,es])

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()


    # make predictions (scaled)
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)

    # invert predictions (back to original)
    if (normalization == 'AN'):
        # originals
        X_trainp, X_testp3, Y_trainp, Y_testp3 = nn_an_den(X_train, X_test, Y_train, Y_test,
                                                           scaler, shift_train, shift_test)
        # predicted
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_an_den(X_train, X_test, trainPredict, testPredict,
                                                                            scaler, shift_train, shift_test)
        print(len(X_trainp))
    if (normalization == 'SW'):
        X_trainp, X_testp3, Y_trainp, Y_testp3 = nn_sw_den(X_train, X_test, Y_train, Y_test,
                                                           scaler_train, scaler_test)
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_sw_den(X_train, X_test, trainPredict, testPredict,
                                                                            scaler_train, scaler_test)
        print(len(X_trainp))
    if (normalization == 'MM'):
        X_trainp, X_testp3, Y_trainp, Y_testp3 = nn_mm_den(X_train, X_test, Y_train, Y_test,
                                                           scaler)
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_mm_den(X_train, X_test, trainPredict, testPredict,
                                                                            scaler)

    if (normalization == 'ZS'):
        X_trainp, X_testp3, Y_trainp, Y_testp3 = nn_zs_den(X_train, X_test, Y_train, Y_test,
                                                           scaler)
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_zs_den(X_train, X_test, trainPredict, testPredict,
                                                                            scaler)

    if (normalization == 'DS'):
        X_trainp, X_testp3, Y_trainp, Y_testp3 = nn_ds_den(X_train, X_test, Y_train, Y_test,
                                                           maximum)
        X_trainp3, X_testp3, new_train_predicted, new_predicted = nn_ds_den(X_train, X_test, trainPredict, testPredict,
                                                                            maximum)

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(new_train_predicted, Y_trainp))
    #trainScore = mean_squared_error(trainPredict, Y_train)
    #print('Train Score: %f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(new_predicted, Y_testp))
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

    # normalizations = ['AN', 'SW', 'MM', 'ZS', 'DS']
    normalizations = ['AN', 'SW']

    with open("output/%d_layers/compare.csv" % n_layers, "a") as fp:
        fp.write("-BTC/MLP-Multi NN\n")

    for normalization in normalizations:
    #for f in range(1,2):
        name='tanh'
        model = Sequential()

        model.add(Dense(12, input_shape = (TRAIN_SIZE, EMB_SIZE),
                kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation(name))

        for l in range(n_layers):
            model.add(Dense(12, input_shape = (TRAIN_SIZE, EMB_SIZE)))
            model.add(Activation(name))
        
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation(name))
        #model.summary()

        trainScore, testScore, epochs, optimizer = evaluate_model(model, name, n_layers,nb_epoch, normalization)
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

