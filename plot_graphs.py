import matplotlib.pylab as plt
import pandas
from matplotlib.pylab import sqrt
import numpy as np
from datetime import datetime


fig_width_pt = 326.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'font.size': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)
# Generate data


dataframe = pandas.read_csv('minidolar/wdo.csv', sep = '|',  engine='python', decimal='.',header=0)

# series = pandas.Series(dataframe['fechamento'].values, index=dataframe['ts'])
# y = np.array(dataframe['fechamento'].tolist())
# # Plot data
# plt.figure(1)
# plt.clf()
# plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
# series[0:100].plot(label='WDOU16')
# #plt.plot(y,'-b')
# plt.xlabel('t')
# plt.ylabel('MINI Dolar')
# plt.legend()
#
# plt.savefig('plots/minidolar.eps')
#


y = dataframe['fechamento']

ewm5 = y.ewm(span=5, min_periods=5).mean()
ewm21 = y.ewm(span=21, min_periods=21).mean()


pandas.Series(dataframe['fechamento'].values, index=dataframe['ts'])
plt.figure(2)
plt.clf()
plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
plt.plot(y[0:160])
plt.plot(ewm5[0:160])
plt.plot(ewm21[0:160])
plt.xlabel('t')
plt.ylabel('MINI Dolar')
plt.legend(['original', 'EMA5', 'EMA21'])

plt.savefig('plots/emas.eps')


#USD-BRL
# dataframe = pandas.read_csv('compare_dolar.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])
#
#
# dataframe = dataframe.sort_values(['w','normalization','k'])
#
# df_an = dataframe.loc[dataframe['normalization'] == 'AN']
# df_sw = dataframe.loc[dataframe['normalization'] == 'SW']
#
# for w in range(2,16):
#     plt.plot(range(2,30), df_an.loc[dataframe['w'] == w]['test'], 'bo')
#     plt.plot(range(2,30), df_sw.loc[dataframe['w'] == w]['test'], 'ro')
#     plt.legend(['AN','SW'])
#     plt.title('Model RMSE with Window size '+str(w)+ ' for MINI Dolar')
#     plt.ylabel('loss')
#     plt.xlabel('K')
#     plt.show()


# for w in range(2,16):
#     plt.plot(range(2,30), df_an.loc[dataframe['w'] == w]['train'], 'bo')
#     plt.plot(range(2,30), df_sw.loc[dataframe['w'] == w]['train'], 'ro')
#     plt.legend(['AN','SW'])
#     plt.title('Model Train RMSE with Window size '+str(w)+ ' for MINI Dolar')
#     plt.ylabel('loss')
#     plt.xlabel('K')
#     plt.show()




#BTC
#dataframe = pandas.read_csv('compare_btc.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', #'epochs'])


#dataframe = dataframe.sort_values(['w','normalization','k'])

#df_an = dataframe.loc[dataframe['normalization'] == 'AN']
#df_sw = dataframe.loc[dataframe['normalization'] == 'SW']

#for w in range(2,16):
#    plt.plot(range(2,30), df_an.loc[dataframe['w'] == w]['test'], 'bo')
#    plt.plot(range(2,30), df_sw.loc[dataframe['w'] == w]['test'], 'ro')
#    plt.legend(['AN','SW'])
#    plt.title('Model RMSE with Window size '+str(w)+ ' for BTC')
#    plt.ylabel('loss')
#    plt.xlabel('K')
#    plt.show()
