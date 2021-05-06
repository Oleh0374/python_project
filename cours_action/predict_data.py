#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 22:43:25 2021

@author: olegnazarenko
"""


# import bibliotheques# Prediction cours de fermeture action en utilisant LSTM (Long Short Term Memory)
from colorama import Fore
from colorama import Style
from loguru import logger
import math
import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime
import pandas as dataframe

# parametrage affichage graphique
plt.style.use('fivethirtyeight')


@logger.catch
def predict_data(datasetfile, date_start, date_end, settest, param):
    datetime.datetime.today()
    day = datetime.datetime.today().strftime('%Y-%m-%d')

    # Recuperation des donnees sur web
    df = web.DataReader(datasetfile, data_source='yahoo', start=date_start, end=date_end)

    # Visualisation des données
    print("Tableau des données:")
    print(df)

    # Creation dataframe avec données de cloture
    data = df.filter(['Close'])

    # Conversion du dataframe vers  numpy
    dataset = data.values

    # Envois des données vers Modele d'entrainement ,prendre nombre des lognes pour entrainement
    # (ici entranement à 80% des données en utilisant longueur des donnees)
    training_data_len = math.ceil(len(dataset) * param)
    #datalen = math.ceil(len(dataset))
    # set des donnes pour modele test
    #settest = datalen - training_data_len
    #print("Period test :" + str(settest))
    #print("Nombre des lignes dans data set choisié :" + str(datalen))
    # impression du nombred des données d'entrainement
    #print("Nombre des lignes pour entrainement :" + str(training_data_len))


    # Mise à l'echelle des données.normalisation avant passer les donnees au reseau des neurons.
    # les donnees seron arrangées selon les parametre de mise à l'echelle entre 0 et 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # scaled_data

    # Creation dataset d'entrainement
    # Dataset mise à l'echelle; de 0 à
    train_data = scaled_data[0:training_data_len, :]
    # diviser les donnes en x_train et y_train
    x_train = []
    y_train = []

    for i in range(settest, len(train_data)):
        # settest = nb derniers valeurs
        x_train.append(train_data[i - settest:i, 0])
        y_train.append(train_data[i, 0])
        # nb de passages
        if i <= settest:
            print(x_train)
            print(y_train)
            print()

    # Convertion x_train et y_train vers numpy arrays pour entrainer LSTM model
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Remodelisation des données pour LSTM qui attends des data en 3 dimentions

    # remodelage a- nb lignes, 2-set entrainement
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print(x_train.shape)

    # Creation du modele LSTM 1 couche à 50 neurons, 2 couche 50, Danse 25 et 1
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compliation du modele
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entrainement du model avec parametrage de passages dans le reseau des neurons
    model.fit(x_train, y_train, batch_size=1, epochs=3)

    # Creation dataset de test
    # Creation du champ de X à Y
    test_data = scaled_data[training_data_len - settest:, :]
    # Creation dataset x et y
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(settest, len(test_data)):
        x_test.append(test_data[i - settest:i, 0])

    # Conversion data vers numpy
    x_test = np.array(x_test)

    # Remodelisation des données
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Modele de prediction des valeurs
    predictions = model.predict(x_test)
    # +denormalisation des donnees
    predictions = scaler.inverse_transform(predictions)

    # Obtention erreur moyenne quadratique, precision avec laquelle modele predit la reponse (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    print(rmse)
    # si valeur n'est pas à 0 le model effectue la prediction correctement

    # Affichage preix reelle et predite
    # print(valid)

    # predictions
    titre_quote = web.DataReader(datasetfile, data_source='yahoo', start=date_start, end=date_end)
    # nouvelle dataframe
    new_df = titre_quote.filter(['Close'])
    # Derniers 60 jours
    period_days = new_df[-settest:].values
    # normalisation data
    period_days_scalled = scaler.transform(period_days)
    # Creation du list
    X_test = list([])
    # Ajout 60 jours
    X_test.append(period_days_scalled)
    # conversion vers numpy
    X_test = np.array(X_test)
    # Modelisation donnees
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # Prix mise à l'echelle (normalisé)
    pred_price = model.predict(X_test)
    # Denormalisation data
    pred_price = scaler.inverse_transform(pred_price)
    print(Fore.BLUE + "Prix de titre predité :  " + datasetfile + Style.RESET_ALL)
    print(pred_price)

    # prix actuelle
    titre_quoteToday = web.DataReader(datasetfile, data_source='yahoo', start=date_end, end=day)
    print(Fore.GREEN+ "Prix de titre à la fermeture reelle pour la date du jour" + Style.RESET_ALL)
    print(titre_quoteToday['Close'])

    # Creation graphique
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    print(valid)

    # Visualisation des données
    plt.figure(figsize=(13, 5))
    plt.title('Prediction pour ' + datasetfile)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Prix de fermeture USD ($)', fontsize=11)
    plt.plot(train['Close'], linewidth=1)
    plt.plot(valid[['Close', 'Predictions']], linewidth=1)
    plt.legend(['Entrainement', 'Valides', 'Predictions'], loc='lower right')
    plt.show()

    return

# prediction à partir d'un fichier des données csv
@logger.catch
def predict_data_file(datasetfile, settest, param):
    datetime.datetime.today()

    # Recuperation des donnees sur web
    df = dataframe.read_csv(datasetfile)
    # Visualisation des données
    #print("Tableau des données:")
    #print(df)

    # Creation dataframe avec données de cloture
    data = df.filter(['Close'])

    # Conversion du dataframe vers champs numpy
    dataset = data.values

    # Envois des données vers Modele d'entrainement ,prendre nombre des lognes pour entrainement
    # (ici entranement à 80% des données en utilisant longueur des donnees)
    training_data_len = math.ceil(len(dataset) * param)

    # Mise à l'echelle des données.normalisation avan passer les donnees au reseau des neurons.
    # les donnees seront arrangées selon les parametre de mise à l'echelle entre 0 et 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # scaled_data

    # Creation dataset d'entrainement
    # Dataset mise à l'echelle; de 0 à
    train_data = scaled_data[0:training_data_len, :]
    # diviser les donnes en x_train et y_train
    x_train = []
    y_train = []

    for i in range(settest, len(train_data)):
        # settest = nb derniers valeurs prises pour entreinement
        x_train.append(train_data[i - settest:i, 0])
        y_train.append(train_data[i, 0])
        # nb de passages
        if i <= settest:
            print(x_train)
            print(y_train)
            print()

    # Convertion x_train et y_train vers numpy arrays pour entrainer LSTM model
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Remodelisation des données pour LSTM qui attends des data en 3 dimentions

    # remodelage a- nb lignes, 2-set entrainement
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print(x_train.shape)

    # Creation du modele LSTM 1 couche à 50 neurons, 60 passages , 2 couche 50, Danse 25 et 1
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compliation du modele
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entrainement du model avec parametrage de passages dans le reseau des neurons
    model.fit(x_train, y_train, batch_size=1, epochs=3)

    # Creation dataset de test
    # Creation du champ de X à Y
    test_data = scaled_data[training_data_len - settest:, :]
    # Creation dataset x et y
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(settest, len(test_data)):
        x_test.append(test_data[i - settest:i, 0])

    # Conversion data vers numpy
    x_test = np.array(x_test)

    # Remodelisation des données
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Modele de prediction des valeurs
    predictions = model.predict(x_test)
    # +denormalisation des donnees
    predictions = scaler.inverse_transform(predictions)

    # Creation graphique
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    print(valid)

    # Visualisation des données
    plt.figure(figsize=(13, 5))
    plt.title('Prediction pour ' + datasetfile)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Prix de fermeture USD ($)', fontsize=11)
    plt.plot(train['Close'], linewidth=1)
    plt.plot(valid[['Close', 'Predictions']], linewidth=1)
    plt.legend(['Entrainement', 'Valides', 'Predictions'], loc='lower right')
    plt.show()

    return