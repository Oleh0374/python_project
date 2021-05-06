#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:05:19 2021

@author: olegnazarenko
"""
#import pip
#pip.main(['install', 'Sequential'])
# import bibliotheques
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
import tensorflow
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Recuperation des donnees sur web
df = web.DataReader('TSLA', data_source='yahoo', start='2021-01-01', end='2021-04-30')
# Visualisation des données
print(df)