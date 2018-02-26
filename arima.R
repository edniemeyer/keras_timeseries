library(TSPred)

dolar = read.csv('minidolar/wdo.csv', sep = '|')
btc = read.csv('btc-usd.csv', sep = ',')

dolar_train = dolar$fechamento[1:round(0.8*nrow(dolar))]
dolar_test = dolar$fechamento[round(0.8*nrow(dolar)+1):nrow(dolar)]

btc_train = btc$close[1:round(0.8*nrow(btc))]
btc_test = btc$close[round(0.8*nrow(btc)+1):nrow(btc)]

xx = dolar$fechamento
currentIndex<-length(dolar_train);
len = NROW( dolar );
forecasts_arima = dolar_test;
repeat
{
  nextIndex = currentIndex + 1
  
  # Get the series
  yy = xx[1:currentIndex]
  
  
  # Save the forecast
  forecasts_arima[currentIndex-length(dolar_train)+1] = arimapred(yy,n.ahead=1)[1]
  
  
  if( nextIndex > len -1) break
  
  currentIndex = nextIndex
  
}


MSE.arima <- MSE(dolar_test,forecasts_arima)
RMSE.arima <- sqrt(MSE.arima)

print(RMSE.arima)
