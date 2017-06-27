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
autocorrelation_plot(dataframe)
plt.show()


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
