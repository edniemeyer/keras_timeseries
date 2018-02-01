import numpy as np
import pandas as pd
from sklearn import preprocessing

#decimal normalization

def decimalNormalize(x):
  return (x / 10^np.log10(max(x)))

#dolar_decimal <- as.data.frame(lapply(csv_dolar_puro,decimalNormalize))
#ibov_decimal <- as.data.frame(lapply(csv_ibov_puro,decimalNormalize))


#decimal denormalize

def decimalDenormalize (x,maxvec):
  return (x*(10^log10(maxvec)))

#den_dolar_decimal <- as.data.frame(Map(decimalDenormalize,dolar_decimal,dolarmaxvec))

def minMaxNormalize(x):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(x)
    return scaler.transform(x), scaler

def minMaxDenormalize(x, scaler):
    return scaler.inverse_transform(x)