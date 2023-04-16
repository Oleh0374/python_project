#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 22:43:25 2021
Updated April 16 2023

FR:
Choix des parametres:
Pourcentage indique la quantite des données utilisés pour entrainement du model de prediction
Plus cette quentité est importente plus le resultat est precise.
EN:
Choice of parameters:
Percentage value used for training the prediction model.
The higher this value, the more accurate the result.


@author: olegnazarenko
"""
import sys, os
from colorama import Fore
from colorama import Style
from loguru import logger
from cours_action.hist_cours import hist_cours
from cours_action.predict_data import predict_data, predict_data_file


#@logger.catch  # trassage des erreurs
# interface client menu d'operations

# Fonction menu

def menu_main():
    os.system('clear')

    print(Fore.BLUE + "Menu :" + Style.RESET_ALL)
    print(Fore.GREEN + "1. Show stock market action - historical data in real time" + Style.RESET_ALL)
    print(Fore.GREEN + "2. Prediction stock market action - close price in real time" + Style.RESET_ALL)
    print(Fore.GREEN + "3. Prediction stock market action - close price from a offline dataset (csv file must be downloaded" + Style.RESET_ALL)
    print(Fore.GREEN + "0. Exit#" + Style.RESET_ALL)
    choice = input("Choix :")
    exec_menu(choice)

    return


#@logger.catch
def exec_menu(choice):
    os.system('clear')
    ch = choice.lower()
    if ch == '':
        menu_actions['menu_main']()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print("Selection invalide. \n")
        menu_actions['menu_main']()
        return


# Prediction sur periode
#@logger.catch
def titre_pred():
    print(Fore.YELLOW + "Prediction of stock market action for a certain period" + Style.RESET_ALL)
    print(Fore.CYAN + "For make a choice of an stock market action here some exemples ( you have to see this #name from a stock market): TSLA - Tesla"+Style.RESET_ALL)
    print(Fore.CYAN + "AAPL - Apple; AIR.PA - AirBus"+Style.RESET_ALL)
    titre = input(Fore.CYAN + "Select the action:" + Style.RESET_ALL)
    print("=========================")
    print(Fore.YELLOW + " Select data volume for prediction" + Style.RESET_ALL)
    date_start = input(Fore.BLUE+"Select start date (format AAAA-MM-JJ) : "+ Style.RESET_ALL)
    date_end = input(Fore.LIGHTGREEN_EX+"Select end date (format AAAA-MM-JJ) : "+ Style.RESET_ALL)
    print("==========================")
    period = int(input(Fore.CYAN + "Select a period for prediction (number of days)" + Style.RESET_ALL))
    print("==========================")
    param = float(input(
        Fore.CYAN + "Select percent value for model traning ( de 0.1 à 0.99 ). \n Valeur "
                    "optimale 80% (0.8):" + Style.RESET_ALL))

    # datasetfile = pd.read_csv(file_path1)

    predict_data(titre, date_start, date_end, period, param)

    print("4. Back to main menu.")
    print("0. Exit.")
    choice = input(">>")
    exec_menu(choice)
    return

# Prediction sur prix d'ouverture
@logger.catch
def cours_today():
    print(Fore.YELLOW + "Shows the  historic graph of close market (in red) (in green)" + Style.RESET_ALL)
    print("For make a choice here some exemples ( you have to see name of action from a stock market): TSLA - Tesla")
    print("AAPL - Apple; AIR.PA - AirBus")
    titre = input(Fore.CYAN + "Select an action:" + Style.RESET_ALL)
    date_start = input(
        Fore.CYAN + "Select start date (format AAAA-MM-JJ):" + Style.RESET_ALL)
    date_end = input(
        Fore.CYAN + "Select end date (format AAAA-MM-JJ):" + Style.RESET_ALL)

    hist_cours(titre, date_start, date_end)

    print("4. Back to main menu.")
    print("0. Exit.")
    choice = input(">>")
    exec_menu(choice)
    return


def data_set():
    print(Fore.YELLOW + "Prediction for a periode from offline dataset" + Style.RESET_ALL)

    datasetfile = input(Fore.CYAN + "Select a CSV file, drug&drop path here :" + Style.RESET_ALL)
    print("=========================")

    period = int(input(Fore.CYAN + "Select a number of days for prediction" + Style.RESET_ALL))
    print("==========================")
    param = float(input(
        Fore.CYAN + "( ( de 0.1 à 0.99 ). \n Value "
                    "optimum 80% (0.8):" + Style.RESET_ALL))

    predict_data_file(datasetfile, period, param)

    print("4. Back to main menu.")
    print("0. Exit.")
    choice = input(">>")
    exec_menu(choice)
    return

# retour au menu
def back():
    menu_actions['menu_main']()


# sortie
def exit():
    sys.exit()


# choix action
menu_actions = {
    'menu_main': menu_main,
    '1': cours_today,
    '2': titre_pred,
    '3': data_set,
    '4': back,
    '0': exit,
}
# Main
if __name__ == "__main__":
    # Launch main menu
    menu_main()
