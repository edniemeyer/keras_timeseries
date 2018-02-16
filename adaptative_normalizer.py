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

X, Y = split_into_chunks(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
X, Y = np.array(X), np.array(Y)

X_train_, X_test_, Y_train_, Y_test_ = create_Xt_Yt(X, Y, percentage=0.80)

X_train, X_test, Y_train, Y_test, scaler, shift_train, shift_test = nn_an(dataset, ewm_dolar, TRAIN_SIZE,TARGET_TIME, LAG_SIZE)

X_train2, X_test2, Y_train2, Y_test2, scaler_train2, scaler_test2 = nn_sw(dataset,TRAIN_SIZE,TARGET_TIME, LAG_SIZE)


X_trainp2, X_testp2, Y_trainp2, Y_testp2 = nn_an_den(X_train, X_test, Y_train, Y_test, scaler, shift_train, shift_test)

X_trainp3, X_testp3, Y_trainp3, Y_testp3 = nn_sw_den(X_train2, X_test2, Y_train2, Y_test2, scaler_train2, scaler_test2)

# X_train, X_test, Y_train, Y_test, maximum = nn_ds(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE)
#
# X_train, X_test, Y_train, Y_test = nn_ds_den(X_train, X_test, Y_train, Y_test, maximum)

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