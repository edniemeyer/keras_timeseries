from pandas import read_csv
from pandas import datetime
from pandas import DataFrame

import math
import pandas as pd

from processing import *

from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA

#dataframe = read_csv('ibov_google_15jun2017_1min_15d.csv', sep = ',', usecols=[1],
#  engine='python', skiprows=8, decimal='.',header=None)

dataframe = read_csv('minidolar/wdo.csv', sep = '|', usecols=[5],  engine='python', decimal='.',header=0)
dataframe = dataframe['fechamento']

start_time = time.time()


# test_stationarity(dataframe)
#
# plt.plot(dataframe.values)
# autocorrelation_plot(dataframe)
# plt.show()


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

p=1
d=1
q=0

WINDOW = 30
STEP = 1
FORECAST = 1

predictions,test = [],[]
for i in range(int(0.8*len(dataframe)), len(dataframe), STEP):
	try:
		x_i = np.asarray(dataframe[0:i]) #ARIMA have to use all timeseries
		y_i = dataframe[i+FORECAST]
		model = ARIMA(x_i, order=(p,d,q))
		model_fit = model.fit(disp=0)
		output = model_fit.forecast()
		yhat = output[0][0]
	except Exception as e:
		break
	
	predictions.append(yhat)
	test.append(y_i)

error = math.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % error)
elapsed_time = (time.time() - start_time)
with open("output/arima.csv", "a") as fp:
	fp.write("p %i, d %i, q %i, rmse %f --%s seconds\n" % (p, d, q, error, elapsed_time))
# plot
#plt.plot(test)
#plt.plot(predictions, color='red')
#plt.show()