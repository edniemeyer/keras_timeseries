from datetime import datetime
import pandas
from processing import *



TRAIN_SIZE = 30
TARGET_TIME = 1
LAG_SIZE = 1
EMB_SIZE = 1

#USD-BRL
dataframe = pandas.read_csv('minidolar/wdo.csv', sep = '|',  engine='python', decimal='.',header=0)
dataset = dataframe['fechamento'].tolist()
media  = dataframe['media'].tolist()

#BTC-USD
btc = pandas.read_csv('btc-usd.csv', sep = ',',  engine='python', decimal='.',header=0)
dataset_btc = btc['close'].tolist()

#print(dataset_btc)

X, Y = split_into_chunks(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=True)
X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y, percentage=0.5)

dados = X_train, X_test, Y_train, Y_test

Xp, Yp = split_into_chunks(dataset, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
Xp, Yp = np.array(Xp), np.array(Yp)
X_trainp, X_testp, Y_trainp, Y_testp = create_Xt_Yt(Xp, Yp, percentage=0.5)

dadosp = X_trainp, X_testp, Y_trainp, Y_testp