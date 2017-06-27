from pandas import read_csv
from pandas import datetime
from pandas import DataFrame

import pandas as pd

from processing import *

from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA

dataframe = read_csv('ibov_google_15jun2017_1min_15d.csv', sep = ',', usecols=[1],
  engine='python', skiprows=8, decimal='.',header=None)


#test_stationarity(dataframe)

#plt.plot(dataset)
#autocorrelation_plot(dataframe)
#plt.show()


# model = ARIMA(dataframe.as_matrix(), order=(5,1,0))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())
# # plot residual errors
# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# plt.show()
# residuals.plot(kind='kde')
# plt.show()
# print(residuals.describe())



X = dataframe.values
size = int(len(X) * 0.9)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()