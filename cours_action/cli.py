#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 22:43:25 2021

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
    print(Fore.GREEN + "1. Affichage historique du cours d'un titre boursier en temps reel" + Style.RESET_ALL)
    print(Fore.GREEN + "2. Prediction prix de cloture du tittre boursier en temps reel" + Style.RESET_ALL)
    print(Fore.GREEN + "3. Prediction prix de cloture du tittre boursier à partir d'un dataset" + Style.RESET_ALL)
    print(Fore.GREEN + "0. Quitter" + Style.RESET_ALL)
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
    print(Fore.YELLOW + "Prediction sur un periode" + Style.RESET_ALL)
    print(Fore.CYAN + "Pour choisir le titre voici quelques exemples: TSLA - Tesla"+Style.RESET_ALL)
    print(Fore.CYAN + "AAPL - Apple; AIR.PA - AirBus"+Style.RESET_ALL)
    titre = input(Fore.CYAN + "Veillez choisir le titre:" + Style.RESET_ALL)
    print("=========================")
    print(Fore.YELLOW + " Selection du volume de données pour la prediction" + Style.RESET_ALL)
    date_start = input(Fore.BLUE+"Pour dataset veillez choisir date de debut (sous forme AAAA-MM-JJ) : "+ Style.RESET_ALL)
    date_end = input(Fore.LIGHTGREEN_EX+"Pour dataset veillez choisir date de fin (sous forme AAAA-MM-JJ) : "+ Style.RESET_ALL)
    print("==========================")
    period = int(input(Fore.CYAN + "Veillez choisir le periode de prediction (en nombre des jours)" + Style.RESET_ALL))
    print("==========================")
    param = float(input(
        Fore.CYAN + "Veillez choisir le pourcentage des donnees pour entrainer le model ( de 0.1 à 0.99 ). \n Valeur "
                    "optimale 80% (0.8):" + Style.RESET_ALL))

    # datasetfile = pd.read_csv(file_path1)

    predict_data(titre, date_start, date_end, period, param)

    print("4. Retour au menu principal.")
    print("0. Quitter.")
    choice = input(">>")
    exec_menu(choice)
    return

# Prediction sur prix d'ouverture
@logger.catch
def cours_today():
    print(Fore.YELLOW + "Affichage historique de fermeture (rouge)  et ouverture (vert)" + Style.RESET_ALL)
    print("Pour choisir le tytre voici quelques exemples: TSLA - Tesla")
    print("AAPL - Apple; AIR.PA - AirBus")
    titre = input(Fore.CYAN + "Veillez choisir le titre:" + Style.RESET_ALL)
    date_start = input(
        Fore.CYAN + "Veillez choisir la date de debut de periode (sous format AAAA-MM-JJ):" + Style.RESET_ALL)
    date_end = input(
        Fore.CYAN + "Veillez choisir la date du fin de periode (sous format AAAA-MM-JJ):" + Style.RESET_ALL)

    hist_cours(titre, date_start, date_end)

    print("4. Retour au menu principal.")
    print("0. Quitter.")
    choice = input(">>")
    exec_menu(choice)
    return


def data_set():
    print(Fore.YELLOW + "Prediction sur un periode à partir dataset" + Style.RESET_ALL)

    datasetfile = input(Fore.CYAN + "Veillez choisir le fichier CSV, copier-coller path ici :" + Style.RESET_ALL)
    print("=========================")

    period = int(input(Fore.CYAN + "Veillez choisir le periode de prediction (en nombre des jours)" + Style.RESET_ALL))
    print("==========================")
    param = float(input(
        Fore.CYAN + "Veillez choisir le pourcentage des donnees pour entrainer le model ( de 0.1 à 0.99 ). \n Valeur "
                    "optimale 80% (0.8):" + Style.RESET_ALL))

    predict_data_file(datasetfile, period, param)

    print("4. Retour au menu principal.")
    print("0. Quitter.")
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
