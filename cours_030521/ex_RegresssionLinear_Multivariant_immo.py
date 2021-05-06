import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np
#import heartrate
#heartrate.trace(browser=True)
# charger le fichier data
df = pd.read_csv("/Users/olegnazarenko/PycharmProjects/cours_210421/dataset/ex_RegressionLinear_Multivariant_immo.csv",
                 delimiter=";")
# Récupérer le prix : les valeurs observées pour la variable Cible
Y = df["prix"]
# Récupérer les variables prédictives : La superficie et le nb chambre
X = df[['taille_en_pieds_carre', 'nb_chambres']]
# Normalisation des features (Features Scaling) : les valeurs seront approximativement comprises entre -1 et 1.
# La Normalisation est utile quand les ordres de grandeur des valeurs des features sont tres différents 
# Ex : Taille d'une maison en "pieds²" est de quelques miliers
# alors que le nombre de chambre est généralement plus petit que 10
scale = StandardScaler()
# X_scaled = scale.fit_transform(X[['taille_en_pieds_carre', 'nb_chambres']].as_matrix())
X_scaled = scale.fit_transform(X[['taille_en_pieds_carre', 'nb_chambres']].to_numpy())
# print(X_scaled)
# OLS : Ordinary Least Squared : une méthode de regression pour estimer une variable cible
# Note : ici que X comporte nos deux variables prédictives
est = sm.OLS(Y, X).fit()
print(est.summary())


# # fonction pour prédiction
# def predict_prix(taille_maison, nb_chambre):
#     return 140.8611 * taille_maison + 1.698e+04 * nb_chambre  # voir est.summary()
#
#
# # test
# print("Le prix estimé pour 4500 pieds & 5 pièces : ", predict_prix(4500, 5))

# version 2 : on récupère directement les valeurs d'estimation --> mieux
print(est.params)
coef1 = est.params.iloc[0]
coef2 = est.params.iloc[1]


# fonction pour prédiction
def predict_prix(taille_maison, nb_chambre):
    return coef1 * taille_maison + coef2 * nb_chambre


# test
print("Le prix estimé pour 4500 pieds & 5 pièces : ", predict_prix(4500, 5))
