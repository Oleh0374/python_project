#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 22:43:25 2021

@author: olegnazarenko
"""
# Prediction cours de fermeture action en utilisant LSTM (Long Short Term Memory)

# import bibliotheques
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

#parametrage affichage graphique
plt.style.use('fivethirtyeight')

# Recuperation des donnees sur web
df = web.DataReader('TSLA', data_source='yahoo', start='2019-01-01', end='2021-04-30')
# Visualisation des données
print("Tableau des données:")
print(df)

# Visualisation historique du prix de fermeture
plt.figure(figsize=(16, 8))
plt.title('Historique prix de fermeture')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Prix € (€)', fontsize=18)
plt.show()

# Visualisation historique du prix d'ouverture'
plt.figure(figsize=(16, 8))
plt.title("Historique prix d'ouverture")
plt.plot(df['Open'], color='red')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Open Prix € (€)', fontsize=18)
plt.show()

# Creation dataframe avec données de cloture
data = df.filter(['Close'])

# Conversion du dataframe vers champs numpy
dataset = data.values

# Envois des données vers Modele d'entrainement ,prendre nombre des lognes pour entrainement
# (ici entranement à 80% des données en utilisant longueur des donnees)
training_data_len = math.ceil(len(dataset) * .8)

# impression du nombred des données d'entrainement
print("Nombre des lignes pour entrainement (-1):")

# cette valeur calculer automatiquement  selon le pourcentage definit et placer en variable pour entrainement du modele
print(training_data_len)

# Mise à l'echelle des données.normalisation avan passer les donnees au reseau des neurons.
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

for i in range(60, len(train_data)):
    # 60 derniers valeurs
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])
    # nb de passages
    if i <= 60:
        print(x_train)
        print(y_train)
        print()

# Convertion x_train et y_train vers numpy arrays pour entrainer LSTM model
x_train, y_train = np.array(x_train), np.array(y_train)

# Remodelisation des données pour LSTM qui attends des data en 3 dimentions

# remodelage a- nb lignes, 2-set entrainement 60
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
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Creation dataset de test
# Creation du champ de X à Y
test_data = scaled_data[training_data_len - 60:, :]
# Creation dataset x et y
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

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

# Creation graphique
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualisation des données
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Prix de fermeture Euro (€)', fontsize=12)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Entrainement', 'Valides', 'Predictions'], loc='lower right')
plt.show()

# Affichage preix reelle et predite
print(valid)

# predictions
tesla_quote = web.DataReader('TSLA', data_source='yahoo', start='2019-01-01', end='2021-05-04')
# nouvelle dataframe
new_df = tesla_quote.filter(['Close'])
# Derniers 60 jours
last_60_days = new_df[-60:].values
# normalisation data
last_60_days_scalled = scaler.transform(last_60_days)
# Creation du list
X_test = []
# Ajout 60 jours
X_test.append(last_60_days_scalled)
# conversion vers numpy
X_test = np.array(X_test)
# Modelisation donnees
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Prix mise à l'echelle (normalisé)
pred_price = model.predict(X_test)
# Denormalisation data
pred_price = scaler.inverse_transform(pred_price)
print("Prix predite pour la date du jour")
print(pred_price)

# prix actuelle
tesla_quoteToday = web.DataReader('TSLA', data_source='yahoo', start='2021-05-04', end='2021-05-04')
print("Prix fermeture reelle pour la date du jour")
print(tesla_quoteToday['Close'])

