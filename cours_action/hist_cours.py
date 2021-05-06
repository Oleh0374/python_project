#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 22:43:25 2021

@author: olegnazarenko
"""
# Prediction cours de fermeture action en utilisant LSTM (Long Short Term Memory)

# import bibliotheques

# import math

import pandas_datareader as web
import matplotlib.pyplot as plt
import datetime
from loguru import logger

# parametrage affichage graphique

plt.style.use('fivethirtyeight')


@logger.catch
def hist_cours(datasetfile, date_start, date_end):
    datetime.datetime.today()

    # Recuperation des donnees sur web
    df = web.DataReader(datasetfile, data_source='yahoo', start=date_start, end=date_end)
    # Visualisation des données
    #print("Tableau des données:")
    #print(df)

    # Visualisation historique du prix de fermeture
    plt.figure(figsize=(14, 6))
    plt.title('Historique prix du prix :' + datasetfile)
    plt.plot(df['Close'], linewidth=1, color='red')
    plt.plot(df['Open'], lw=1, color='green')
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Close Prix USD ($)', fontsize=11)
    plt.legend(['Close', 'Open'], loc='upper right')
    plt.show()

    return
