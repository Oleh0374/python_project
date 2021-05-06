import matplotlib.pyplot as plt
import numpy as np
#import math
#import pandas as pd
#from pandas import pandas_datareader as web
# import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
plt.style.use('fivethirtyeight')
from loguru import logger
from itertools import combinations


@logger.catch
def predict_period(dataset, period, param):
    NbLines = dataset.shape[0]

    # period prediction sur un period chisié

    PrixOrg = dataset["Close"].iloc[(NbLines - period):]

    ##Code1 : Pour montrer qq traitements de DataFrame
    dataset = dataset.drop(["High"], axis=1)  # supprimer la colonne Volume, car non nécessaire
    dataset = dataset.drop(["Low"], axis=1)
    dataset["label"] = dataset["Close"].shift(
        -period)  # ajouter une colonne label pour stocker la prédict : décaler avec Close/Last


    print(dataset.head)

    Data = dataset.drop(["label"], axis=1)  # on crée les données dans dataframe Data (Close/Last, Open, High, Low)
    # !!!on garder df

    X = Data.values  # récupère les valeurs

    X = preprocessing.scale(X)  # scale : normaliser les données

    print(X)

    X = X[:-period]

    # print(X)
    dataset.dropna(inplace=True)

    # print(df)
    Target = dataset.label

    Y = Target.values  # récupère les valeurs de colonne Label et les mettre en Y

    # print(Y)
    # print(np.shape(X), np.shape(Y))  # les dimensions de données
    # modèle LinearRegression
    # créer les données de train, de test...
    # (maths/stat, etc. ???)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=param)
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    lr.score(X_test, Y_test)

    # print("R^2",X_test,Y_test)

    # prédiction
    X_predict = X[-period:]
    Forecast = lr.predict(X_predict)

    print("Prédiction de valeur de ", period, " jours : ", Forecast)

    # graphe : comparer donnees Org (red) et Prediction (blue)
    Ypredict = np.array(Forecast)
    X = np.arange(period)  # axis X
    plt.plot(X, Ypredict)
    YOrg = np.array(PrixOrg)
    plt.plot(X, YOrg, color='red')
    plt.show()

@logger.catch

def predict_open(dataset, day, param):

    NbLines = dataset.shape[0]

    # period prediction sur un period choisié

    PrixOrg = dataset["Open"].iloc[(NbLines - day):]

    ##Code1 : Pour montrer qq traitements de DataFrame
    dataset = dataset.drop(["High"], axis=1)  # supprimer la colonne Volume, car non nécessaire
    dataset = dataset.drop(["Low"], axis=1)
    dataset = dataset.drop(["Adj Close"], axis=1)
    dataset = dataset.drop(["Volume"], axis=1)
    dataset["label"] = dataset["Open"].shift(-day)  # ajouter une colonne label pour stocker la prédict : décaler avec Close/Last

    print(dataset.head)

    Data = dataset.drop(["label"], axis=1)  # on crée les données dans dataframe Data (Close/Last, Open, High, Low)
    # !!!on garder df

    X = Data.values  # récupère les valeurs

    X = preprocessing.scale(X)  # scale : normaliser les données

    #print(X)
    X = X[:-day]

    #print(X)
    dataset.dropna(inplace=True)

    # print(df)
    Target = dataset.label

    Y = Target.values  # récupère les valeurs de colonne Label et les mettre en Y

    # print(Y)
    # print(np.shape(X), np.shape(Y))  # les dimensions de données
    # modèle LinearRegression
    # créer les données de train, de test...
    # (maths/stat, etc. ???)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=param)
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    lr.score(X_test, Y_test)

    # print("R^2",X_test,Y_test)

    # prédiction
    X_predict = X[:-day]
    Forecast = lr.predict(X_predict)

    print("Prédiction de prix d'ouverture ", Forecast)

    # graphe : comparer donnees Org (red) et Prediction (blue)
    Ypredict = np.array(Forecast)
    X = np.arange(day)  # axis X
    plt.plot(X, Ypredict)
    YOrg = np.array(PrixOrg)
    plt.plot(X, YOrg, color='red')
    plt.show()
