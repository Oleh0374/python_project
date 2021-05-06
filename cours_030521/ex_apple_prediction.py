# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:07:26 2021
@author: Nguyen Minh Duc
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from loguru import logger
from itertools import combinations

# import tensorflow as tf

# charger & traiter les données d'origine
#df = pd.read_csv('/Users/olegnazarenko/PycharmProjects/cours_210421/dataset/apple_1a.csv')
df = pd.read_csv('/Users/olegnazarenko/PycharmProjects/cours_210421/dataset/apple_1a.csv')
# qq traitements pour faciliter le travail
df = df.replace({'\$': ''}, regex=True)
df = df.astype({"Close/Last": float})
df = df.astype({"Open": float})
df["Date"] = pd.to_datetime(df.Date, format="%m/%d/%Y")
df = df.set_index("Date")
print(df)
print("Strucutre et ex. de données : ",df.head)
print(df.size)      #nb total de données
print(df.shape[0])  # nb de lignes
print(df.shape[1])  # nb de colonnes

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

# 2ème partie : prédiction
## note : il y a 251 lignes dans le dataset
## pour illustrer, on fait une prédiction pour 30 jours en se basant sur 221 lignes de données (251-30)
## le reste 30 lignes dans les données d'origine pour comparer entre réel et modèle prédictif


# colonne Close/Last original : pour comparer
NbLines = df.shape[0]

num = 30  # prédiction pour 30 jours
PrixOrg = df["Close/Last"].iloc[(NbLines - num):]

##Code1 : Pour montrer qq traitements de DataFrame
df = df.drop(["Volume"], axis=1)  # supprimer la colonne Volume, car non nécessaire
df["label"] = df["Close/Last"].shift(-num)  # ajouter une colonne label pour stocker la prédict : décaler avec Close/Last
# print(df.head)

Data = df.drop(["label"], axis=1)  # on crée les données dans dataframe Data (Close/Last, Open, High, Low)
# !!!on garder df
X = Data.values  # récupère les valeurs
X = preprocessing.scale(X)  # scale : normaliser les données
# print(X)
X = X[:-num]
# print(X)
df.dropna(inplace=True)
# print(df)
Target = df.label
Y = Target.values  # récupère les valeurs de colonne Label et les mettre en Y
# print(Y)
# print(np.shape(X), np.shape(Y))  # les dimensions de données
# modèle LinearRegression
# créer les données de train, de test...
# (maths/stat, etc. ???)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
lr = LinearRegression()
lr.fit(X_train, Y_train)
lr.score(X_test, Y_test)
# print("R^2",X_test,Y_test)
# prédiction
X_predict = X[-num:]
Forecast = lr.predict(X_predict)
print("Prédiction de valeur de ", num, " jours : ", Forecast)
# graphe : comparer donnees Org (red) et Prediction (blue)
Ypredict = np.array(Forecast)
X = np.arange(num)  # axis X
plt.plot(X, Ypredict)
YOrg = np.array(PrixOrg)
plt.plot(X, YOrg, color='red')
plt.show()