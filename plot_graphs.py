import matplotlib.pylab as plt
import pandas
from matplotlib.pylab import sqrt, boxplot
import math
import numpy as np
from processing import *
from datetime import datetime


fig_width_pt = 496.0  # Get this from LaTeX using \showthe\columnwidth
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

#minidolar
# dataframe = pandas.read_csv('minidolar/wdo.csv', sep = '|',  engine='python', decimal='.',header=0)
#
# series = pandas.Series(dataframe['fechamento'].values, index=dataframe['ts'])
# y = np.array(dataframe['fechamento'].tolist())
# # Plot data
# plt.figure(1)
# plt.clf()
# plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
# # series[0:100].plot(label='WDOU16')
# plt.plot(y,'-b')
# plt.axvline(x=int(0.7*len(y)), color='k')
# plt.text(int(0.6*len(y)), max(y), 'train')
# plt.text(int(0.72*len(y)), max(y), 'val')
# plt.axvline(x=int(0.8*len(y)), color='k')
# plt.text(int(0.82*len(y)), max(y), 'test')
# plt.xlabel('t')
# plt.ylabel('MINI Dolar')
# plt.legend(['WDOU16'])
#
# plt.savefig('plots/minidolar.eps')


#variacoes minidolar
dataframe = pandas.read_csv('minidolar/wdo.csv', sep = '|',  engine='python', decimal='.',header=0)
dataset_original = dataframe['fechamento']

dataset_original =  [100 * (b - a) / a for a, b in zip(dataset_original[::1], dataset_original[1::1])]


dataset_original = pandas.DataFrame(dataset_original)
dataset_original = dataset_original[0]


y = np.array(dataset_original.tolist())
# Plot data
plt.figure(1)
plt.clf()
plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
# series[0:100].plot(label='WDOU16')
plt.plot(y,'-b')
plt.axvline(x=int(0.7*len(y)), color='k')
plt.text(int(0.6*len(y)), max(y), 'train')
plt.text(int(0.72*len(y)), max(y), 'val')
plt.axvline(x=int(0.8*len(y)), color='k')
plt.text(int(0.82*len(y)), max(y), 'test')
plt.xlabel('t')
plt.ylabel('Variações MINI Dolar')
#plt.legend(['WDOU16'])

plt.savefig('plots/variacoes.eps')



#btc
# dataframe = pandas.read_csv('btc-usd.csv', sep=',', engine='python', decimal='.', header=0)
#
# y = np.array(dataframe['close'].tolist())
# # Plot data
# plt.figure(1)
# plt.clf()
# plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
# plt.plot(y,'-b')
# plt.axvline(x=int(0.7*len(y)), color='k')
# plt.text(int(0.6*len(y)), max(y), 'train')
# plt.text(int(0.72*len(y)), max(y), 'val')
# plt.axvline(x=int(0.8*len(y)), color='k')
# plt.text(int(0.82*len(y)), max(y), 'test')
# plt.xlabel('t')
# plt.ylabel('Bitcoin')
# plt.legend(['BTC(USD)'])
#
# plt.savefig('plots/btc.eps')


#rainfall
# dataframe = pandas.read_csv('annual-rainfall-at-fortaleza-bra.csv', sep=',', engine='python', header=0)
#
# y = np.array(dataframe['rainfall'].tolist())
# # Plot data
# plt.figure(1)
# plt.clf()
# plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
# plt.plot(y,'-b')
# plt.axvline(x=int(0.7*len(y)), color='k')
# plt.text(int(0.6*len(y)), max(y), 'train')
# plt.text(int(0.72*len(y)), max(y), 'val')
# plt.axvline(x=int(0.8*len(y)), color='k')
# plt.text(int(0.82*len(y)), max(y), 'test')
# plt.xlabel('t')
# plt.ylabel('Chuvas em Fortaleza (mm)')
# #plt.legend(['BTC(mm)'])
#
# plt.savefig('plots/rainfall.eps')
#

#furnas
# dataframe = pandas.read_csv('furnas-vazoes-medias-mensais-m3s.csv', sep=',', engine='python', header=0)
#
# y = np.array(dataframe['furnas'].tolist())
# # Plot data
# plt.figure(1)
# plt.clf()
# plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
# plt.plot(y,'-b')
# plt.axvline(x=int(0.7*len(y)), color='k')
# plt.text(int(0.6*len(y)), max(y), 'train')
# plt.text(int(0.72*len(y)), max(y), 'val')
# plt.axvline(x=int(0.8*len(y)), color='k')
# plt.text(int(0.82*len(y)), max(y), 'test')
# plt.xlabel('t')
# plt.ylabel('Vazões médias em FURNAS (m³/s)')
# #plt.legend(['BTC(mm)'])
#
# plt.savefig('plots/furnas.eps')


#EMAS
# y = dataframe['fechamento']
#
# ewm5 = y.ewm(span=5, min_periods=5).mean()
# ewm21 = y.ewm(span=21, min_periods=21).mean()
#
#
# plt.figure(2)
# plt.clf()
# plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
# plt.plot(y[0:160])
# plt.plot(ewm5[0:160])
# plt.plot(ewm21[0:160])
# plt.xlabel('t')
# plt.ylabel('MINI Dolar')
# plt.legend(['original', 'EMA5', 'EMA21'])
#
# plt.savefig('plots/emas.eps')
#

#SW exemplo


# y = dataframe['fechamento']
#
# hv, hv_scaler = minMaxNormalize(y[120:140].values.reshape(-1,1))
# lv, lv_scaler= minMaxNormalize(y[50:70].values.reshape(-1,1))
#
# plt.figure(3)
# plt.clf()
# plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
# plt.plot(range(0,20),lv)
# plt.plot(range(30,50),hv)
# plt.legend(['SW #1', 'SW #2'])
# plt.savefig('plots/sw_exemplo.eps')


#AN exemplo


#
# y = dataframe['fechamento']
# ewm5 = y.ewm(span=5, min_periods=5).mean()
#
# y = np.array(y.iloc[5 - 1:])
# ewm5 = np.array(ewm5.iloc[5 - 1:])
#
# X, Y, shift = split_into_chunks_adaptive_type(y[0:540], ewm5[0:540], 20, 1, 1, binary=False,
#                                              scale=True, type='o')
# X, Y, shift = np.array(X), np.array(Y), np.array(shift)
#
# serie, scaler = minMaxNormalize((X[0:540]).reshape(-1,1))
#
#
# hv = minMaxNormalizeOver((X[116]).reshape(-1,1), scaler)
# lv = minMaxNormalizeOver((X[46]).reshape(-1,1), scaler)
#
#
# plt.figure(4)
# plt.clf()
# plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
# plt.plot(range(0,20),lv)
# plt.plot(range(30,50),hv)
# plt.legend(['AN #1', 'AN #2'])
# plt.savefig('plots/an_exemplo.eps')



#MM exemplo


# y = dataframe['fechamento']
# y_1d = y[0:540]
#
# sample, scaler = minMaxNormalize(y_1d[0:90].values.reshape(-1,1))
# y_mm = minMaxNormalizeOver(y_1d.values.reshape(-1,1), scaler)
#
# plt.figure(5)
# plt.clf()
# plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
# plt.plot(y_mm)
# plt.xlabel('t(min)')
# plt.savefig('plots/mm_exemplo.eps')
#

#Dec

# y = dataframe['fechamento']
# y_1d = y[0:540]
#
# maximum = max(y_1d[0:90].values.reshape(-1))
#
# sample = decimalNormalize(y_1d[0:90].values.reshape(-1,1))
# y_dec = decimalNormalizeOver(y_1d.values.reshape(-1,1), maximum)
#
# plt.figure(6)
# plt.clf()
# plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
# plt.plot(y_dec)
# plt.plot(np.ones(540), 'r')
# plt.xlabel('t(min)')
# plt.savefig('plots/dec_exemplo.eps')
#
#
# y_1d_new = decimalDenormalize(y_dec, maximum)


#z-score

# y = dataframe['fechamento']
# y_1d = y[0:540]
#
# sample, scaler = zNormalize(y_1d[0:90].values.reshape(-1,1))
# y_z = zNormalizeOver(y_1d.values.reshape(-1,1), scaler)
#
# y_z_total, sca =  zNormalize(y_1d.values.reshape(-1,1))
# sca.mean_
# np.sqrt(sca.var_)
#
# plt.figure(7)
# plt.clf()
# plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
# plt.plot(y_z)
# plt.plot(y_z_total)
# plt.legend([(r'Z #1 ($\mu =%.2f, \sigma =%.2f$)'% (scaler.mean_,np.sqrt(scaler.var_))), (r'Z #2 ($\mu =%.2f, \sigma =%.2f$)' %(sca.mean_,np.sqrt(sca.var_)))])
# plt.xlabel('t(min)')
# plt.savefig('plots/z_exemplo.eps')
#
#
# y_1d_new = zDenormalize(y_z, scaler)




# Fs = 100
# f = 5
# sample = 100
# x = np.arange(sample)
# y = np.sin(2 * np.pi * f * x / Fs)
# ewm5 = pandas.DataFrame(y).ewm(span=5, min_periods=5).mean()
#
# plt.figure(8)
# plt.clf()
# plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
# plt.plot(y)
# plt.plot(ewm5)
# plt.plot(np.zeros(len(y)), ':k')
# plt.xlabel('t')
# plt.legend(['sen(t)', 'EMA5'])
# plt.savefig('plots/an_avg_zero_problem.eps')
#




#AN types compare exemplo
#
#
# y = dataframe['fechamento']
# ewm5 = y.ewm(span=5, min_periods=5).mean()
#
# y = np.array(y.iloc[5 - 1:])
# ewm5 = np.array(ewm5.iloc[5 - 1:])
#
# X, Y, shift = split_into_chunks_adaptive_type(y[0:540], ewm5[0:540], 20, 1, 1, binary=False,
#                                              scale=True, type='o')
# X, Y, shift = np.array(X), np.array(Y), np.array(shift)
#
#
# X2, Y2, shift2 = split_into_chunks_adaptive_type(y[0:540], ewm5[0:540], 20, 1, 1, binary=False,
#                                              scale=True, type='c')
#
# X2, Y2, shift2 = np.array(X2), np.array(Y2), np.array(shift2)
#
#
# serie, scaler = minMaxNormalize((X[0:540]).reshape(-1,1))
#
#
# serie2, scaler2 = minMaxNormalize((X2[0:540]).reshape(-1,1))
#
#
# hv = minMaxNormalizeOver((X[116]).reshape(-1,1), scaler)
# lv = minMaxNormalizeOver((X[46]).reshape(-1,1), scaler)
#
# hv2 = minMaxNormalizeOver((X2[116]).reshape(-1,1), scaler2)
# lv2 = minMaxNormalizeOver((X2[46]).reshape(-1,1), scaler2)


# plt.figure(9)
# plt.clf()
# plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
# plt.plot(range(0,20),lv - lv2)
# plt.plot(range(30,50),hv - hv2)
# plt.legend(['AN - ANS #1', 'AN - ANS #2'])
# plt.savefig('plots/an_compare.eps')





#
# #USD-BRL
dataframe = pandas.read_csv('compare_dolar.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])
dataframe = dataframe.sort_values(['w','normalization','k'])

df_an = dataframe.loc[dataframe['normalization'] == 'AN']
df_sw = dataframe.loc[dataframe['normalization'] == 'SW']
df_ano = dataframe.loc[dataframe['normalization'] == 'ANO']
df_anc = dataframe.loc[dataframe['normalization'] == 'ANC']
#
# for w in range(2,16):
#     plt.clf()
#     plt.plot(range(2, 30), df_sw.loc[dataframe['w'] == w]['test'], 'bo')
#     plt.plot(range(2, 30), df_ano.loc[dataframe['w'] == w]['test'], 'ro')
#     plt.plot(range(2, 30), df_an.loc[dataframe['w'] == w]['test'], 'go')
#     plt.plot(range(2, 30), df_anc.loc[dataframe['w'] == w]['test'], 'k+')
#     plt.legend(['SW','AN','ANS', 'ANC'])
#     plt.title('Janela de tamanho w = '+str(w)+ ' para Dataset I')
#     plt.ylabel('RMSE')
#     plt.xlabel('k')
#     plt.savefig('plots/dolar_rmse_k_w_' + str(w) + '.eps')
#
#

print(dataframe[dataframe['test'] == min(dataframe['test'])])

#
#
# #BTC
dataframe = pandas.read_csv('compare_btc.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])


dataframe = dataframe.sort_values(['w','normalization','k'])
#
df_an = dataframe.loc[dataframe['normalization'] == 'AN']
df_sw = dataframe.loc[dataframe['normalization'] == 'SW']
df_ano = dataframe.loc[dataframe['normalization'] == 'ANO']
df_anc = dataframe.loc[dataframe['normalization'] == 'ANC']
#
# for w in range(2,16):
#     plt.clf()
#     plt.plot(range(2, 30), df_sw.loc[dataframe['w'] == w]['test'], 'bo')
#     plt.plot(range(2, 30), df_ano.loc[dataframe['w'] == w]['test'], 'ro')
#     plt.plot(range(2, 30), df_an.loc[dataframe['w'] == w]['test'], 'go')
#     plt.plot(range(2, 30), df_anc.loc[dataframe['w'] == w]['test'], 'k+')
#     plt.legend(['SW','AN','ANS', 'ANC'])
#     plt.title('Janela de tamanho w = '+str(w)+ ' para Dataset II')
#     plt.ylabel('RMSE')
#     plt.xlabel('k')
#     plt.savefig('plots/btc_rmse_k_w_' + str(w) + '.eps')

print(dataframe[dataframe['test'] == min(dataframe['test'])])


#furnas
dataframe = pandas.read_csv('compare_furnas.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])


dataframe = dataframe.sort_values(['w','normalization','k'])

df_an = dataframe.loc[dataframe['normalization'] == 'AN']
df_sw = dataframe.loc[dataframe['normalization'] == 'SW']
df_ano = dataframe.loc[dataframe['normalization'] == 'ANO']
df_anc = dataframe.loc[dataframe['normalization'] == 'ANC']
#
# for w in range(2,16):
#     plt.clf()
#     plt.plot(range(2, 30), df_sw.loc[dataframe['w'] == w]['test'], 'bo')
#     plt.plot(range(2, 30), df_ano.loc[dataframe['w'] == w]['test'], 'ro')
#     plt.plot(range(2, 30), df_an.loc[dataframe['w'] == w]['test'], 'go')
#     plt.plot(range(2, 30), df_anc.loc[dataframe['w'] == w]['test'], 'k+')
#     plt.legend(['SW','AN','ANS', 'ANC'])
#     plt.title('Janela de tamanho w = '+str(w)+ ' para Dataset IV')
#     plt.ylabel('RMSE')
#     plt.xlabel('k')
#     plt.savefig('plots/furnas_rmse_k_w_' + str(w) + '.eps')

print(dataframe[dataframe['test'] == min(dataframe['test'])])

#rainfall
dataframe = pandas.read_csv('compare_rainfall.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])


dataframe = dataframe.sort_values(['w','normalization','k'])

df_an = dataframe.loc[dataframe['normalization'] == 'AN']
df_sw = dataframe.loc[dataframe['normalization'] == 'SW']
df_ano = dataframe.loc[dataframe['normalization'] == 'ANO']
df_anc = dataframe.loc[dataframe['normalization'] == 'ANC']

# for w in range(2,16):
#     plt.clf()
#     plt.plot(range(2, 30), df_sw.loc[dataframe['w'] == w]['test'], 'bo')
#     plt.plot(range(2, 30), df_ano.loc[dataframe['w'] == w]['test'], 'ro')
#     plt.plot(range(2, 30), df_an.loc[dataframe['w'] == w]['test'], 'go')
#     plt.plot(range(2, 30), df_anc.loc[dataframe['w'] == w]['test'], 'k+')
#     plt.legend(['SW','AN','ANS', 'ANC'])
#     plt.title('Janela de tamanho w = '+str(w)+ ' para Dataset III')
#     plt.ylabel('RMSE')
#     plt.xlabel('k')
#     plt.savefig('plots/rain_rmse_k_w_' + str(w) + '.eps')
#


print(dataframe[dataframe['test'] == min(dataframe['test'])])

#
#
#
# dataframe = pandas.read_csv('epochs-ideal.csv', sep = ',',  engine='python', decimal='.')
#
# plt.clf()
# plt.plot(dataframe['epoch'].values, dataframe['val_loss'].values)
# plt.savefig('plots/epochs_val_loss.eps')




#
#
# #rainfall neurons
# dataframe = pandas.read_csv('compare_rainfall-neuron-0layers.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])
#
# w3 = dataframe.loc[dataframe['w'] == 3]['test'].values
# w6 = dataframe.loc[dataframe['w'] == 6]['test'].values
# w9 = dataframe.loc[dataframe['w'] == 9]['test'].values
# w12 = dataframe.loc[dataframe['w'] == 12]['test'].values
#
#
# plt.clf()
# boxplot([w3,w6,w9,w12], positions = [3,6,9,12])
# # plt.title('Rede Neural com uma camada oculta para Dataset III')
# plt.ylabel('RMSE')
# plt.xlabel('neurons')
# plt.savefig('plots/rain_rmse_neuron_0l.eps')
#
#
# dataframe = pandas.read_csv('compare_rainfall-neuron-1layers.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])
#
# w3 = dataframe.loc[dataframe['w'] == 3]['test'].values
# w6 = dataframe.loc[dataframe['w'] == 6]['test'].values
# w9 = dataframe.loc[dataframe['w'] == 9]['test'].values
# w12 = dataframe.loc[dataframe['w'] == 12]['test'].values
#
#
# plt.clf()
# boxplot([w3,w6,w9,w12], positions = [3,6,9,12])
# # plt.title('Rede Neural com duas camadas ocultas para Dataset III')
# plt.ylabel('RMSE')
# plt.xlabel('neurons')
# plt.savefig('plots/rain_rmse_neuron_1l.eps')
#
# #furnas neurons
# dataframe = pandas.read_csv('compare_furnas-neuron-0layers.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])
#
# w3 = dataframe.loc[dataframe['w'] == 3]['test'].values
# w6 = dataframe.loc[dataframe['w'] == 6]['test'].values
# w9 = dataframe.loc[dataframe['w'] == 9]['test'].values
# w12 = dataframe.loc[dataframe['w'] == 12]['test'].values
#
#
# plt.clf()
# boxplot([w3,w6,w9,w12], positions = [3,6,9,12])
# # plt.title('Rede Neural com uma camada oculta para Dataset IV')
# plt.ylabel('RMSE')
# plt.xlabel('neurons')
# plt.savefig('plots/furnas_rmse_neuron_0l.eps')
#
#
# dataframe = pandas.read_csv('compare_furnas-neuron-1layers.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])
#
# w3 = dataframe.loc[dataframe['w'] == 3]['test'].values
# w6 = dataframe.loc[dataframe['w'] == 6]['test'].values
# w9 = dataframe.loc[dataframe['w'] == 9]['test'].values
# w12 = dataframe.loc[dataframe['w'] == 12]['test'].values
#
#
# plt.clf()
# boxplot([w3,w6,w9,w12], positions = [3,6,9,12])
# # plt.title('Rede Neural com duas camadas ocultas para Dataset IV')
# plt.ylabel('RMSE')
# plt.xlabel('neurons')
# plt.savefig('plots/furnas_rmse_neuron_1l.eps')


#dolar neurons
#dataframe = pandas.read_csv('compare_dolar-neuron-0layers.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])

#w3 = dataframe.loc[dataframe['w'] == 3]['test'].values
#w6 = dataframe.loc[dataframe['w'] == 6]['test'].values
#w9 = dataframe.loc[dataframe['w'] == 9]['test'].values
#w12 = dataframe.loc[dataframe['w'] == 12]['test'].values


#plt.clf()
#boxplot([w3,w6,w9,w12], positions = [3,6,9,12])
# plt.title('Rede Neural com uma camada oculta para Dataset IV')
#plt.ylabel('RMSE')
#plt.xlabel('neurons')
#plt.savefig('plots/dolar_rmse_neuron_0l.eps')


#dataframe = pandas.read_csv('compare_dolar-neuron-1layers.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])

#w3 = dataframe.loc[dataframe['w'] == 3]['test'].values
#w6 = dataframe.loc[dataframe['w'] == 6]['test'].values
#w9 = dataframe.loc[dataframe['w'] == 9]['test'].values
#w12 = dataframe.loc[dataframe['w'] == 12]['test'].values


#plt.clf()
#boxplot([w3,w6,w9,w12], positions = [3,6,9,12])
# plt.title('Rede Neural com duas camadas ocultas para Dataset IV')
#plt.ylabel('RMSE')
#plt.xlabel('neurons')
#plt.savefig('plots/dolar_rmse_neuron_1l.eps')


#btc neurons
# dataframe = pandas.read_csv('compare_btc-neuron-0layers.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])
#
# w3 = dataframe.loc[dataframe['w'] == 3]['test'].values
# w6 = dataframe.loc[dataframe['w'] == 6]['test'].values
# w9 = dataframe.loc[dataframe['w'] == 9]['test'].values
# w12 = dataframe.loc[dataframe['w'] == 12]['test'].values
#
#
# plt.clf()
# boxplot([w3,w6,w9,w12], positions = [3,6,9,12])
# # plt.title('Rede Neural com uma camada oculta para Dataset IV')
# plt.ylabel('RMSE')
# plt.xlabel('neurons')
# plt.savefig('plots/btc_rmse_neuron_0l.eps')
#
#
# dataframe = pandas.read_csv('compare_btc-neuron-1layers.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])
#
# w3 = dataframe.loc[dataframe['w'] == 3]['test'].values
# w6 = dataframe.loc[dataframe['w'] == 6]['test'].values
# w9 = dataframe.loc[dataframe['w'] == 9]['test'].values
# w12 = dataframe.loc[dataframe['w'] == 12]['test'].values
#
#
# plt.clf()
# boxplot([w3,w6,w9,w12], positions = [3,6,9,12])
# # plt.title('Rede Neural com duas camadas ocultas para Dataset IV')
# plt.ylabel('RMSE')
# plt.xlabel('neurons')
# plt.savefig('plots/btc_rmse_neuron_1l.eps')



#variacoes neurons
dataframe = pandas.read_csv('compare_variacoes-neuron-0layers.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])

w3 = dataframe.loc[dataframe['w'] == 3]['test'].values
w6 = dataframe.loc[dataframe['w'] == 6]['test'].values
w9 = dataframe.loc[dataframe['w'] == 9]['test'].values
w12 = dataframe.loc[dataframe['w'] == 12]['test'].values


plt.clf()
boxplot([w3,w6,w9,w12], positions = [3,6,9,12])
# plt.title('Rede Neural com uma camada oculta para Dataset V')
plt.ylabel('RMSE')
plt.xlabel('neurons')
plt.savefig('plots/variacoes_rmse_neuron_0l.eps')


dataframe = pandas.read_csv('compare_variacoes-neuron-1layers.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])

w3 = dataframe.loc[dataframe['w'] == 3]['test'].values
w6 = dataframe.loc[dataframe['w'] == 6]['test'].values
w9 = dataframe.loc[dataframe['w'] == 9]['test'].values
w12 = dataframe.loc[dataframe['w'] == 12]['test'].values


plt.clf()
boxplot([w3,w6,w9,w12], positions = [3,6,9,12])
# plt.title('Rede Neural com duas camadas ocultas para Dataset V')
plt.ylabel('RMSE')
plt.xlabel('neurons')
plt.savefig('plots/variacoes_rmse_neuron_1l.eps')



###################################################################################################







#rainfall k
dataframe = pandas.read_csv('compare_rainfall-k.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])

w3 = dataframe.loc[dataframe['w'] == 3]['test'].values.mean()
w8 = dataframe.loc[dataframe['w'] == 8]['test'].values.mean()
w13 = dataframe.loc[dataframe['w'] == 13]['test'].values.mean()
w18 = dataframe.loc[dataframe['w'] == 18]['test'].values.mean()
w23 = dataframe.loc[dataframe['w'] == 23]['test'].values.mean()
w28 = dataframe.loc[dataframe['w'] == 28]['test'].values.mean()



plt.clf()
plt.plot( [3,8,13,18,23,28], [w3,w8,w13,w18,w23,w28],'bo')
# plt.title('Rede Neural com uma camada oculta para Dataset III')
plt.ylabel('RMSE')
plt.xlabel('k')
plt.savefig('plots/rain_rmse_k.eps')

#furnas k
dataframe = pandas.read_csv('compare_furnas-k.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])

w3 = dataframe.loc[dataframe['w'] == 3]['test'].values.mean()
w8 = dataframe.loc[dataframe['w'] == 8]['test'].values.mean()
w13 = dataframe.loc[dataframe['w'] == 13]['test'].values.mean()
w18 = dataframe.loc[dataframe['w'] == 18]['test'].values.mean()
w23 = dataframe.loc[dataframe['w'] == 23]['test'].values.mean()
w28 = dataframe.loc[dataframe['w'] == 28]['test'].values.mean()



plt.clf()
plt.plot([3,8,13,18,23,28], [w3,w8,w13,w18,w23,w28],'bo')
# plt.title('Rede Neural com uma camada oculta para Dataset III')
plt.ylabel('RMSE')
plt.xlabel('k')
plt.savefig('plots/furnas_rmse_k.eps')


#dolar k
dataframe = pandas.read_csv('compare_dolar-k.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])

w3 = dataframe.loc[dataframe['w'] == 3]['test'].values.mean()
w8 = dataframe.loc[dataframe['w'] == 8]['test'].values.mean()
w13 = dataframe.loc[dataframe['w'] == 13]['test'].values.mean()
w18 = dataframe.loc[dataframe['w'] == 18]['test'].values.mean()
w23 = dataframe.loc[dataframe['w'] == 23]['test'].values.mean()
w28 = dataframe.loc[dataframe['w'] == 28]['test'].values.mean()



plt.clf()
plt.plot( [3,8,13,18,23,28], [w3,w8,w13,w18,w23,w28], 'bo')
# plt.title('Rede Neural com uma camada oculta para Dataset III')
plt.ylabel('RMSE')
plt.xlabel('k')
plt.savefig('plots/dolar_rmse_k.eps')


#btc k
dataframe = pandas.read_csv('compare_btc-k.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])

w3 = dataframe.loc[dataframe['w'] == 3]['test'].values.mean()
w8 = dataframe.loc[dataframe['w'] == 8]['test'].values.mean()
w13 = dataframe.loc[dataframe['w'] == 13]['test'].values.mean()
w18 = dataframe.loc[dataframe['w'] == 18]['test'].values.mean()
w23 = dataframe.loc[dataframe['w'] == 23]['test'].values.mean()
w28 = dataframe.loc[dataframe['w'] == 28]['test'].values.mean()



plt.clf()
plt.plot([3,8,13,18,23,28],[w3,w8,w13,w18,w23,w28], 'bo')
# plt.title('Rede Neural com uma camada oculta para Dataset III')
plt.ylabel('RMSE')
plt.xlabel('k')
plt.savefig('plots/btc_rmse_k.eps')


#variacoes k
dataframe = pandas.read_csv('compare_variacoes-k.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])

w3 = dataframe.loc[dataframe['w'] == 3]['test'].values.mean()
w8 = dataframe.loc[dataframe['w'] == 8]['test'].values.mean()
w13 = dataframe.loc[dataframe['w'] == 13]['test'].values.mean()
w18 = dataframe.loc[dataframe['w'] == 18]['test'].values.mean()
w23 = dataframe.loc[dataframe['w'] == 23]['test'].values.mean()
w28 = dataframe.loc[dataframe['w'] == 28]['test'].values.mean()



plt.clf()
plt.plot([3,8,13,18,23,28],[w3,w8,w13,w18,w23,w28], 'bo')
# plt.title('Rede Neural com uma camada oculta para Dataset V')
plt.ylabel('RMSE')
plt.xlabel('k')
plt.savefig('plots/variacoes_rmse_k.eps')



###################################################################################################







#rainfall w
dataframe = pandas.read_csv('compare_rainfall-w.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])

w2 = dataframe.loc[dataframe['w'] == 2]['test'].values.mean()
w7 = dataframe.loc[dataframe['w'] == 7]['test'].values.mean()
w12 = dataframe.loc[dataframe['w'] == 12]['test'].values.mean()
w17 = dataframe.loc[dataframe['w'] == 17]['test'].values.mean()
w22 = dataframe.loc[dataframe['w'] == 22]['test'].values.mean()
w27 = dataframe.loc[dataframe['w'] == 27]['test'].values.mean()



plt.clf()
plt.plot( [2,7,12,17,22,27], [w2,w7,w12,w17,w22,w27],'bo')
# plt.title('Rede Neural com uma camada oculta para Dataset III')
plt.ylabel('RMSE')
plt.xlabel('w')
plt.savefig('plots/rain_rmse_w.eps')

#furnas w
dataframe = pandas.read_csv('compare_furnas-w.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])

w2 = dataframe.loc[dataframe['w'] == 2]['test'].values.mean()
w7 = dataframe.loc[dataframe['w'] == 7]['test'].values.mean()
w12 = dataframe.loc[dataframe['w'] == 12]['test'].values.mean()
w17 = dataframe.loc[dataframe['w'] == 17]['test'].values.mean()
w22 = dataframe.loc[dataframe['w'] == 22]['test'].values.mean()
w27 = dataframe.loc[dataframe['w'] == 27]['test'].values.mean()



plt.clf()
plt.plot([2,7,12,17,22,27], [w2,w7,w12,w17,w22,w27],'bo')
# plt.title('Rede Neural com uma camada oculta para Dataset III')
plt.ylabel('RMSE')
plt.xlabel('w')
plt.savefig('plots/furnas_rmse_w.eps')


#dolar w
dataframe = pandas.read_csv('compare_dolar-w.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])

w2 = dataframe.loc[dataframe['w'] == 2]['test'].values.mean()
w7 = dataframe.loc[dataframe['w'] == 7]['test'].values.mean()
w12 = dataframe.loc[dataframe['w'] == 12]['test'].values.mean()
w17 = dataframe.loc[dataframe['w'] == 17]['test'].values.mean()
w22 = dataframe.loc[dataframe['w'] == 22]['test'].values.mean()
w27 = dataframe.loc[dataframe['w'] == 27]['test'].values.mean()



plt.clf()
plt.plot( [2,7,12,17,22,27], [w2,w7,w12,w17,w22,w27], 'bo')
# plt.title('Rede Neural com uma camada oculta para Dataset III')
plt.ylabel('RMSE')
plt.xlabel('w')
plt.savefig('plots/dolar_rmse_w.eps')


#btc w
dataframe = pandas.read_csv('compare_btc-w.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])

w2 = dataframe.loc[dataframe['w'] == 2]['test'].values.mean()
w7 = dataframe.loc[dataframe['w'] == 7]['test'].values.mean()
w12 = dataframe.loc[dataframe['w'] == 12]['test'].values.mean()
w17 = dataframe.loc[dataframe['w'] == 17]['test'].values.mean()
w22 = dataframe.loc[dataframe['w'] == 22]['test'].values.mean()
w27 = dataframe.loc[dataframe['w'] == 27]['test'].values.mean()



plt.clf()
plt.plot([2,7,12,17,22,27],[w2,w7,w12,w17,w22,w27], 'bo')
# plt.title('Rede Neural com uma camada oculta para Dataset III')
plt.ylabel('RMSE')
plt.xlabel('w')
plt.savefig('plots/btc_rmse_w.eps')


#variacoes w
dataframe = pandas.read_csv('compare_variacoes-w.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])

w2 = dataframe.loc[dataframe['w'] == 2]['test'].values.mean()
w7 = dataframe.loc[dataframe['w'] == 7]['test'].values.mean()
w12 = dataframe.loc[dataframe['w'] == 12]['test'].values.mean()
w17 = dataframe.loc[dataframe['w'] == 17]['test'].values.mean()
w22 = dataframe.loc[dataframe['w'] == 22]['test'].values.mean()
w27 = dataframe.loc[dataframe['w'] == 27]['test'].values.mean()



plt.clf()
plt.plot([2,7,12,17,22,27],[w2,w7,w12,w17,w22,w27], 'bo')
# plt.title('Rede Neural com uma camada oculta para Dataset V')
plt.ylabel('RMSE')
plt.xlabel('w')
plt.savefig('plots/variacoes_rmse_w.eps')





############################################################


#minidolar all norm
dataframe = pandas.read_csv('compare_dolar_all.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])


dataframe = dataframe.sort_values(['w','normalization','k'])

df_ano = dataframe.loc[dataframe['normalization'] == 'ANo']['test'].values.mean()
df_anc = dataframe.loc[dataframe['normalization'] == 'ANc']['test'].values.mean()
df_and = dataframe.loc[dataframe['normalization'] == 'ANd']['test'].values.mean()
df_sw = dataframe.loc[dataframe['normalization'] == 'SW']['test'].values.mean()
df_mm = dataframe.loc[dataframe['normalization'] == 'MM']['test'].values.mean()
df_zs = dataframe.loc[dataframe['normalization'] == 'ZS']['test'].values.mean()
df_ds = dataframe.loc[dataframe['normalization'] == 'DS']['test'].values.mean()

print("minidolar")
print(df_ano)
print(df_anc)
print(df_and)
print(df_sw)
print(df_mm)
print(df_zs)
print(df_ds)


#btc all norm
dataframe = pandas.read_csv('compare_btc_all.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])


dataframe = dataframe.sort_values(['w','normalization','k'])

df_ano = dataframe.loc[dataframe['normalization'] == 'ANo']['test'].values.mean()
df_anc = dataframe.loc[dataframe['normalization'] == 'ANc']['test'].values.mean()
df_and = dataframe.loc[dataframe['normalization'] == 'ANd']['test'].values.mean()
df_sw = dataframe.loc[dataframe['normalization'] == 'SW']['test'].values.mean()
df_mm = dataframe.loc[dataframe['normalization'] == 'MM']['test'].values.mean()
df_zs = dataframe.loc[dataframe['normalization'] == 'ZS']['test'].values.mean()
df_ds = dataframe.loc[dataframe['normalization'] == 'DS']['test'].values.mean()

print("btc")
print(df_ano)
print(df_anc)
print(df_and)
print(df_sw)
print(df_mm)
print(df_zs)
print(df_ds)



#furnas all norm
dataframe = pandas.read_csv('compare_furnas_all.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])


dataframe = dataframe.sort_values(['w','normalization','k'])

df_ano = dataframe.loc[dataframe['normalization'] == 'ANo']['test'].values.mean()
df_anc = dataframe.loc[dataframe['normalization'] == 'ANc']['test'].values.mean()
df_and = dataframe.loc[dataframe['normalization'] == 'ANd']['test'].values.mean()
df_sw = dataframe.loc[dataframe['normalization'] == 'SW']['test'].values.mean()
df_mm = dataframe.loc[dataframe['normalization'] == 'MM']['test'].values.mean()
df_zs = dataframe.loc[dataframe['normalization'] == 'ZS']['test'].values.mean()
df_ds = dataframe.loc[dataframe['normalization'] == 'DS']['test'].values.mean()

print("furnas")
print(df_ano)
print(df_anc)
print(df_and)
print(df_sw)
print(df_mm)
print(df_zs)
print(df_ds)



#rainfall all norm
dataframe = pandas.read_csv('compare_rainfall_all.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])


dataframe = dataframe.sort_values(['w','normalization','k'])

df_ano = dataframe.loc[dataframe['normalization'] == 'ANo']['test'].values.mean()
df_anc = dataframe.loc[dataframe['normalization'] == 'ANc']['test'].values.mean()
df_and = dataframe.loc[dataframe['normalization'] == 'ANd']['test'].values.mean()
df_sw = dataframe.loc[dataframe['normalization'] == 'SW']['test'].values.mean()
df_mm = dataframe.loc[dataframe['normalization'] == 'MM']['test'].values.mean()
df_zs = dataframe.loc[dataframe['normalization'] == 'ZS']['test'].values.mean()
df_ds = dataframe.loc[dataframe['normalization'] == 'DS']['test'].values.mean()

print("rainfall")
print(df_ano)
print(df_anc)
print(df_and)
print(df_sw)
print(df_mm)
print(df_zs)
print(df_ds)



#variacoes all norm
dataframe = pandas.read_csv('compare_variacoes_all.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])


dataframe = dataframe.sort_values(['w','normalization','k'])

df_ano = dataframe.loc[dataframe['normalization'] == 'ANo']['test'].values.mean()
df_anc = dataframe.loc[dataframe['normalization'] == 'ANc']['test'].values.mean()
df_and = dataframe.loc[dataframe['normalization'] == 'ANd']['test'].values.mean()
df_sw = dataframe.loc[dataframe['normalization'] == 'SW']['test'].values.mean()
df_mm = dataframe.loc[dataframe['normalization'] == 'MM']['test'].values.mean()
df_zs = dataframe.loc[dataframe['normalization'] == 'ZS']['test'].values.mean()
df_ds = dataframe.loc[dataframe['normalization'] == 'DS']['test'].values.mean()

print("variacoes")
print(df_ano)
print(df_anc)
print(df_and)
print(df_sw)
print(df_mm)
print(df_zs)
print(df_ds)
