from datetime import datetime

from gdax import GDAX


def btc_usd_1min(start, end):
  return GDAX('BTC-USD').fetch(start, end, 1)



if __name__ == "__main__":
  data_frame = btc_usd_1min(datetime(2017, 12, 1), datetime(2018, 12, 31))
  print(data_frame)

  # Save to CSV.
data_frame.to_csv('btc-usd.csv')