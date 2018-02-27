import matplotlib.pylab as plt
import pandas


#USD-BRL
dataframe = pandas.read_csv('compare_dolar.csv', sep = ',',  engine='python', decimal='.', header = None, names=['w', 'k', 'activation', 'normalization', 'train', 'test', 'optimizer', 'epochs'])


dataframe = dataframe.sort_values(['w','normalization','k'])

df_an = dataframe.loc[dataframe['normalization'] == 'AN']
df_sw = dataframe.loc[dataframe['normalization'] == 'SW']

for w in range(2,16):
    plt.plot(range(2,30), df_an.loc[dataframe['w'] == w]['test'], 'bo')
    plt.plot(range(2,30), df_sw.loc[dataframe['w'] == w]['test'], 'ro')
    plt.legend(['AN','SW'])
    plt.title('Model RMSE with Window size '+str(w)+ ' for MINI Dolar')
    plt.ylabel('loss')
    plt.xlabel('K')
    plt.show()


# for w in range(2,16):
#     plt.plot(range(2,30), df_an.loc[dataframe['w'] == w]['train'], 'bo')
#     plt.plot(range(2,30), df_sw.loc[dataframe['w'] == w]['train'], 'ro')
#     plt.legend(['AN','SW'])
#     plt.title('Model Train RMSE with Window size '+str(w)+ ' for MINI Dolar')
#     plt.ylabel('loss')
#     plt.xlabel('K')
#     plt.show()