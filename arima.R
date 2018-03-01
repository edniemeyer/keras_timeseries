# install.packages("TSPred")
library(TSPred)


# MINI Dollar

start_time <- Sys.time()

dolar = read.csv('minidolar/wdo.csv', sep = '|')
dolar_train = dolar$fechamento[1:round(0.8*nrow(dolar))]
dolar_test = dolar$fechamento[round(0.8*nrow(dolar)+1):nrow(dolar)]

xx = dolar$fechamento
currentIndex<-length(dolar_train);
len = NROW( dolar );
forecasts_arima_fit = dolar_test;
repeat
{
  nextIndex = currentIndex + 1

  # Get the series
  yy = xx[1:currentIndex]
  yp = c(xx[nextIndex])

  fittest <- fittestArima(yy, yp)

  # Save the forecast
  forecasts_arima_fit[currentIndex-length(dolar_train)+1] = fittest$pred$pred[1]


  if( nextIndex > len -1) break

  currentIndex = nextIndex
}


MSE.arima <- MSE(dolar_test,forecasts_arima_fit)
RMSE.arima <- sqrt(MSE.arima)
print('MINI Dollar')
print(RMSE.arima)

end_time <- Sys.time()

elapsed_time = end_time - start_time

writeLines(paste('MINI Dollar',RMSE.arima,elapsed_time, sep=','), file='output/arimaR.csv')

# BTC-USD


btc = read.csv('btc-usd.csv', sep = ',')
btc_train = btc$close[1:round(0.8*nrow(btc))]
btc_test = btc$close[round(0.8*nrow(btc)+1):nrow(btc)]

xx = btc$close
currentIndex<-length(btc_train);
len = NROW( btc );
forecasts_arima_fit = btc_test;

repeat
{
  nextIndex = currentIndex + 1

  # Get the series
  yy = xx[1:currentIndex]
  yp = c(xx[nextIndex])

  fittest <- fittestArima(yy, yp)

  # Save the forecast
  forecasts_arima_fit[currentIndex-length(btc_train)+1] = fittest$pred$pred[1]


  if( nextIndex > len -1) break

  currentIndex = nextIndex
}


MSE.arima <- MSE(btc_test,forecasts_arima_fit)
RMSE.arima <- sqrt(MSE.arima)
print('BTC-USD')
print(RMSE.arima)

end_time <- Sys.time()

elapsed_time = end_time - start_time

writeLines(paste('BTC-USD',RMSE.arima,elapsed_time, sep=','), file='output/arimaR.csv')
