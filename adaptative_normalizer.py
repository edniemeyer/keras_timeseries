from datetime import datetime
import pandas
from processing import *



TRAIN_SIZE = 30
TARGET_TIME = 1
LAG_SIZE = 1
EMB_SIZE = 1

#USD-BRL
dataframe = pandas.read_csv('minidolar/wdo.csv', sep = '|',  engine='python', decimal='.',header=0)
dataset = dataframe['fechamento']
media  = dataframe['media'].tolist()

ewm_dolar = dataset.ewm(span=5, min_periods=5).mean()


#removendo NaN
dataset = dataset.iloc[4:]
ewm_dolar = ewm_dolar.iloc[4:]

X_train, X_test, Y_train, Y_test, maximum = nn_ds(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)

X_train, X_test, Y_train, Y_test = nn_ds_den(X_train, X_test, Y_train, Y_test, maximum)

#import matplotlib.pyplot as plt
#plt.plot(dataset_norm)

#BTC-USD
btc = pandas.read_csv('btc-usd.csv', sep = ',',  engine='python', decimal='.',header=0)
dataset_btc = btc['close']
ewm_btc = dataset_btc.ewm(span=5, min_periods=5).mean()


#removendo NaN
dataset_btc = dataset_btc.iloc[4:]
ewm_btc = ewm_btc.iloc[4:]

#print(dataset_btc)

#
# nn_sw(dataset, TRAIN_SIZE,TARGET_TIME,LAG_SIZE)
#
# X_train, X_test, Y_train, Y_test, scaler_train, scaler_test, shift_train, shift_test = nn_an(dataset, ewm_dolar, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)
#
#
# X_train, X_test, Y_train, Y_test = nn_an_den(X_train, X_test, Y_train, Y_test, scaler_train, scaler_test, shift_train, shift_test)