from datetime import datetime
import pandas

from normalizer import minMaxNormalize, minMaxNormalizeOver
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


#minmax normalization without sliding windows

# X, Y = split_into_chunks(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
# X, Y = np.array(X), np.array(Y)
# X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y, percentage=0.35)
#
# X_normalizado, scaler = minMaxNormalize(X_train.tolist())

train, test = create_Train_Test(dataset, 0.35)
train_normalizado, scaler2 = minMaxNormalize(train.values.reshape(-1,1))

dataset_norm = minMaxNormalizeOver(dataset.values.reshape(-1,1), scaler2)

X, Y = split_into_chunks(dataset_norm.reshape(-1), TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y, percentage=0.35)

#import matplotlib.pyplot as plt
#plt.plot(dataset_norm)


#minmax normalization with sliding windows

X, Y = split_into_chunks(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y, percentage=0.35)

X_normalizado, scaler = minMaxNormalize(X_train.tolist())



#BTC-USD
btc = pandas.read_csv('btc-usd.csv', sep = ',',  engine='python', decimal='.',header=0)
dataset_btc = btc['close']
ewm_btc = dataset_btc.ewm(span=5, min_periods=5).mean()


#removendo NaN
dataset_btc = dataset_btc.iloc[4:]
ewm_btc = ewm_btc.iloc[4:]

#print(dataset_btc)

X, Y, shift = split_into_chunks_adaptive(dataset, ewm_dolar, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=True)
X, Y, shift = np.array(X), np.array(Y), np.array(shift)
X_train, X_test, Y_train, Y_test, shift_train, shift_test = create_Xt_Yt_adaptive(X, Y, shift, percentage=0.35)

dados = X_train, X_test, Y_train, Y_test

