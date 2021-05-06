import pandas as pd
#import pip
#pip.main(['install', 'pandas_datareader'])
import pandas_datareader as web
# import tensorflow as tf
# charger & traiter les données d'origine
from arc.predict_data_old import predict_period, predict_open
from loguru import logger


@logger.catch
def data_trait1(dataset, period, param):
    # nettoyage

    #dataset = dataset.replace({'\$': ''}, regex=True)
    dataset["Date"] = pd.to_datetime(dataset.Date, format="%Y/%m/%d")
    dataset = dataset.set_index("Date")
    dataset = dataset.astype({"Open": float})
    dataset = dataset.astype({"High": float})
    dataset = dataset.astype({"Low": float})
    dataset = dataset.astype({"Close": float})
    dataset = dataset.astype({"Adj Close": float})
    print(dataset)
    print("Strucutre et ex. de données : ", dataset.head)
    print(dataset.size)  # nb total de données
    print(dataset.shape[0])  # nb de lignes
    print(dataset.shape[1])  # nb de colonnes

    predict_period(dataset, period, param)
    return


@logger.catch
def data_trait2(dataset, day, param):

    #dataset from web

    df = web.DataReader('TSLA', data_source='yahoo', start='2021-01-01', end='2021-04-30')
    #print(df)
    # nettoyage

    # dataset = dataset.replace({'\$': ''}, regex=True)
    dataset["Date"] = pd.to_datetime(dataset.Date, format="%Y/%m/%d")
    dataset = dataset.set_index("Date")
    dataset = dataset.astype({"Open": float})
    #dataset = dataset.astype({"High": float})
    #dataset = dataset.astype({"Low": float})
    #dataset = dataset.astype({"Close": float})
    #dataset = dataset.astype({"Adj Close": float})
    print(dataset)
    #print("Strucutre et ex. de données : ", dataset.head)
    print(dataset.size)  # nb total de données
    print(dataset.shape[0])  # nb de lignes
    print(dataset.shape[1])  # nb de colonnes

    predict_open(dataset, day, param)
    return

# #1ere partie : exploiter le dataset actuel
# #max Volume
# maxVol=df['Volume'].max()
# print("Volume max : ",maxVol)
# #max Volume avec d'autres info
# maxVolID=df['Volume'].idxmax()
# print("Infos avec volume max : ",df.loc[maxVolID])
# #graphe de prix d'ouverture
# Yprix=np.array(df['Close/Last'])
# X=np.arange(df.shape[0]) # axis X
# plt.plot(X,Yprix)
# plt.show()
