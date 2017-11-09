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
EMB_SIZE = 4 #numero de features

train = pd.read_csv('minidolar/train.csv', sep = ',',  engine='python', decimal='.',header=0)
test = pd.read_csv('minidolar/test.csv', sep = ',',  engine='python', decimal='.',header=0)

train_shift = train['shift']
train_target = train['f0']
train_open = train[['v0','v4','v8','v12','v16','v20','v24','v28','v32','v36','v40','v44','v48','v52','v56','v60','v64','v68','v72','v76','v80','v84','v88','v92','v96','v100','v104','v108','v112','v116']]
train_high = train[['v1','v5','v9','v13','v17','v21','v25','v29','v33','v37','v41','v45','v49','v53','v57','v61','v65','v69','v73','v77','v81','v85','v89','v93','v97','v101','v105','v109','v113','v117']]
train_low = train[['v2','v6','v10','v14','v18','v22','v26','v30','v34','v38','v42','v46','v50','v54','v58','v62','v66','v70','v74','v78','v82','v86','v90','v94','v98','v102','v106','v110','v114','v118']]
train_close = train[['v3','v7','v11','v15','v19','v23','v27','v31','v35','v39','v43','v47','v51','v55','v59','v63','v67','v71','v75','v79','v83','v87','v91','v95','v99','v103','v107','v111','v115','v119']]

test_shift = test['shift']
test_target = test['f0']
test_open = test[['v0','v4','v8','v12','v16','v20','v24','v28','v32','v36','v40','v44','v48','v52','v56','v60','v64','v68','v72','v76','v80','v84','v88','v92','v96','v100','v104','v108','v112','v116']]
test_high = test[['v1','v5','v9','v13','v17','v21','v25','v29','v33','v37','v41','v45','v49','v53','v57','v61','v65','v69','v73','v77','v81','v85','v89','v93','v97','v101','v105','v109','v113','v117']]
test_low = test[['v2','v6','v10','v14','v18','v22','v26','v30','v34','v38','v42','v46','v50','v54','v58','v62','v66','v70','v74','v78','v82','v86','v90','v94','v98','v102','v106','v110','v114','v118']]
test_close = test[['v3','v7','v11','v15','v19','v23','v27','v31','v35','v39','v43','v47','v51','v55','v59','v63','v67','v71','v75','v79','v83','v87','v91','v95','v99','v103','v107','v111','v115','v119']]


def evaluate_model(model, name, n_layers, ep):
    X_train, X_test, Y_train, Y_test =  np.column_stack((train_open.values,train_high.values,train_low.values,train_close.values)),  np.column_stack((test_open.values,test_high.values,test_low.values,test_close.values)),  np.array(train_target.values.reshape(train_target.size,1)),  np.array(test_target.values.reshape(test_target.size,1))
    X_trainp, X_testp, Y_trainp, Y_testp = X_train+train_shift.values.reshape(train_shift.size,1), X_test+test_shift.values.reshape(test_shift.size,1), Y_train+train_shift.values.reshape(train_shift.size,1), Y_test + test_shift.values.reshape(test_shift.size,1)


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
    X_train = np.reshape(X_train, (X_train.shape[0], int(X_train.shape[1]/EMB_SIZE), EMB_SIZE))
    X_test = np.reshape(X_test, (X_test.shape[0], int(X_test.shape[1]/EMB_SIZE), EMB_SIZE))
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
    new_predicted = testPredict+test_shift.values.reshape(test_shift.size,1)
    new_train_predicted= trainPredict+train_shift.values.reshape(train_shift.size,1)

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
        fp.write("-MINIDOLAR/Convolutional-Multi NN\n")

    hals = []
    #data_original = pd.read_csv('./data/AAPL1216.csv')[::-1]
    #data_original = pd.read_csv('ibov_google_15jun2017_1min_15d.csv', sep = ',',  engine='python', skiprows=8, decimal='.',header=None)

    WINDOW = 30
    TRAIN_SIZE=WINDOW
    
    STEP = 1
    FORECAST = 1

    
    for f in range(23,24):
            #name=Hyperbolic(rho=0.9)
            name='relu'
            model = Sequential()

            #model.add(Dense(500, input_shape = (TRAIN_SIZE, )))
            #model.add(Activation(name))

            model.add(Conv1D(input_shape = (TRAIN_SIZE, EMB_SIZE),filters=15,kernel_size=f,activation=name,padding='causal',strides=1,
                    kernel_regularizer=regularizers.l2(0.01)))
            #model.add(MaxPooling1D(pool_size=2))
            for l in range(n_layers):
                model.add(Conv1D(input_shape = (TRAIN_SIZE, EMB_SIZE),filters=15,kernel_size=f,activation=name,padding='causal',strides=1))
                #model.add(MaxPooling1D(pool_size=1))
            
            model.add(Dropout(0.5))
            model.add(Flatten())

            #model.add(Dense(5))
            #model.add(Dropout(0.25))
            #model.add(Activation(name))
            
            model.add(Dense(1))
            model.add(Activation('linear'))
            #model.summary()

            trainScore, testScore, epochs, optimizer = evaluate_model(model, name, n_layers,nb_epoch)
            # if(testScore_aux > testScore):
            #     testScore_aux=testScore
            #     f_aux = f

            elapsed_time = (time.time() - start_time)
            with open("output/%d_layers/compare.csv" % n_layers, "a") as fp:
                fp.write("%i,%s,%f,%f,%d,%s --%s seconds\n" % (f, name, trainScore, testScore, epochs, optimizer, elapsed_time))
                #fp.write("%s,%f,%f,%d,%s --%s seconds\n" % (name, trainScore, testScore, epochs, optimizer, elapsed_time))
                

            model = None

        #print("melhor parametro: %i" % f_aux)

if __name__ == "__main__":
   __main__(sys.argv[1:])