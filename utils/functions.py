# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 09:28:44 2020

@author: alspe
"""


import os
import glob
import cmath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score



j = complex(0, 1)

def alpha_JCAPL(w, materiau, debug=False):
    """
    Calcule la tortuosité dynamique selon le modèle JCAPL (Johnson-Champoux-Allard-Pride-Lafarge)
    Sources : https://apmr.matelys.com/PropagationModels/MotionlessSkeleton/JohnsonChampouxAllardPrideLafargeModel.html
    
    Entrées :
    w : la pulsation
    materiau : 
    k_0 : la viscosité perméable statique
    phi : la porosité
    alpha_inf : la limite haute fréquence de la tortuosité
    Delta : la longueur de viscosité caractéristique
    alpha_0 : la tortuosité visqueuse statique
    eta : la viscosité dynamique de l'air (1.84 × 10−5 N.s.m−2 à température ambiante)
    
    Retourne :
    alpha : la tortuosité dynamique
    """
    
    # constantes
    M = 8*materiau.k_0*materiau.alpha_inf/(materiau.phi*materiau.Delta**2)
    P = M/(4*(materiau.alpha_0/materiau.alpha_inf-1))
    b = materiau.rho_0*materiau.k_0*materiau.alpha_inf/(materiau.eta*materiau.phi)
    L = 2*P**2/(M*b)
    
    # valeur d'alpha
    alpha = materiau.alpha_inf*(1+1/(j*w*b)*(1-P+P*(1+1/L*j*w)**0.5))
    
    # debug
    if debug:
        print("M : ", M)
        print("P : ", P)
        print("b : ", b)
        print("L : ", L)
    
    return alpha




def correlation(df, pred_df, split=False):
    """
    Calcule le coefficient r2 pour l'ensemble [[Re(z) for z in df] [Im(z) for z in df]]

    Parameters
    ----------
    df : pandas DataFrame
        Les vraies valeurs de la fonction.
    pred_df : pandas DataFrame
        Les valeurs explorées.

    Returns
    -------
    r2coef : float
        Le coefficient r2 calculé.

    """
    
    if split:
        r2coef_r = r2_score(df.alpha_r.values,
                            pred_df.alpha_r.values)
        r2coef_i = r2_score(df.alpha_i.values,
                            pred_df.alpha_i.values)
        return r2coef_r, r2coef_i

    else:
        r2coef = r2_score(df.alpha_r.values      + df.alpha_i.values,
                          pred_df.alpha_r.values + pred_df.alpha_i.values)
        return r2coef




def mse(df, pred_df):
    """
    Calcule l'erreur quadratique moyenne (MSE) pour l'ensemble [[Re(z) for z in df] [Im(z) for z in df]]

    Parameters
    ----------
    df : pandas DataFrame
        Les vraies valeurs de la fonction.
    pred_df : pandas DataFrame
        Les valeurs explorées.

    Returns
    -------
    r2coef : float
        L'erreur quadratique moyenne calculée.

    """
    r2coef = mean_squared_error(df.alpha_r.values      + df.alpha_i.values,
                                pred_df.alpha_r.values + pred_df.alpha_i.values)
    
    return r2coef




def plot(df, pred_df, mode="humain", path_name=None):
    # Réels et Imaginaires selon la pulsation
    plt.figure(figsize=(2*4, 4))
    
    plt.subplot(1, 2, 1)
    plt.xlabel("$\omega$")
    plt.xscale('log')
    plt.ylabel("Re")
    plt.grid(True, linestyle='--')
    plt.scatter(df["w"], df["alpha_r"])
    plt.scatter(pred_df["w"], pred_df["alpha_r"], label="pred")
    plt.title("$R^2$=%.2f, MSE=%.2f" % (r2_score(df["alpha_r"], pred_df["alpha_r"]), 
                                        mean_squared_error(df["alpha_r"], pred_df["alpha_r"])))
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.xlabel("$\omega$")
    plt.xscale('log')
    plt.ylabel("Im")
    plt.grid(True, linestyle='--')
    plt.scatter(df["w"], df["alpha_i"])
    plt.scatter(pred_df["w"], pred_df["alpha_i"], label="pred")
    plt.title("$R^2$=%.2f, MSE=%.2f" % (r2_score(df["alpha_i"], pred_df["alpha_i"]), 
                                        mean_squared_error(df["alpha_i"], pred_df["alpha_i"])))
    plt.legend()
    
    plt.tight_layout()
    if path_name is not None:
        plt.savefig('results/{0}/fig/reel_w_imag_w.png'.format(path_name))
    if mode=="humain":
        plt.show()
    else:
        plt.close()
    
    
    # Réel suivis d'Imaginaire selon pulsation
    plt.figure(figsize=(8, 8))    
    
    plt.xlabel("$\omega$")
    plt.xscale('log')
    plt.ylabel("Im")
    plt.grid(True, linestyle='--')
    plt.scatter(df["w"].values + df["w"].values[-1]*pred_df["w"].values, 
                df["alpha_r"].values + df["alpha_i"].values)
    plt.scatter(df["w"].values + df["w"].values[-1]*pred_df["w"].values, 
                pred_df["alpha_r"].values + pred_df["alpha_i"].values, 
                label="pred")
    plt.title("$R^2$=%.2f, MSE=%.2f" % (r2_score(df["alpha_r"].values      + df["alpha_i"].values, 
                                                 pred_df["alpha_r"].values + pred_df["alpha_i"].values), 
                                        mean_squared_error(df["alpha_r"].values      + df["alpha_i"].values,
                                                           pred_df["alpha_r"].values + pred_df["alpha_i"].values)))
    plt.legend()
    
    plt.tight_layout()
    if mode=="humain":
        plt.show()
    else:
        plt.close()
    
    
    # Imaginaire selon Réel
    plt.figure(figsize=(8, 8))
    
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.grid(True, linestyle='--')
    plt.scatter(df["alpha_r"], df["alpha_i"])
    plt.scatter(pred_df["alpha_r"], pred_df["alpha_i"], label="pred")
    plt.title("$R^2$=%.2f, MSE=%.2f" % (r2_score(df["alpha_r"].values      + df["alpha_i"].values, 
                                                 pred_df["alpha_r"].values + pred_df["alpha_i"].values), 
                                        mean_squared_error(df["alpha_r"].values      + df["alpha_i"].values,
                                                           pred_df["alpha_r"].values + pred_df["alpha_i"].values)))
    plt.legend()
    
    plt.tight_layout()
    if path_name is not None:
        plt.savefig('results/{0}/fig/reel_imag.png'.format(path_name))
    if mode=="humain":
        plt.show()
    else:
        plt.close()
    

    # Gain selon pulsation et Phase selon pulsation
    y_true = list(cmath.polar(complex(z)) for z in df["alpha"].values)
    y_pred = list(cmath.polar(complex(z)) for z in pred_df["alpha"].values)
    
    plt.figure(figsize=(8, 8))
        
    plt.subplot(2, 1, 1)
    plt.semilogx(df["w"], [r for (r, phi) in y_true])
    plt.semilogx(pred_df["w"], [r for (r, phi) in y_pred], label="pred")
    plt.legend()
    plt.grid(True,linestyle='--')
    plt.xlabel("$\omega$")
    plt.xscale('log')
    plt.title("Magnetude")
    
    plt.subplot(2, 1, 2)
    plt.semilogx(df["w"], [phi*180/np.pi for (r, phi) in y_true])
    plt.semilogx(pred_df["w"], [phi*180/np.pi for (r, phi) in y_pred], label="pred")
    plt.legend()
    plt.grid(True,linestyle='--')
    plt.xlabel("$\omega$")
    plt.xscale('log')
    plt.title("Phase")
        
    plt.tight_layout()
    if path_name is not None:
        plt.savefig('results/{0}/fig/gain_phase.png'.format(path_name))
    if mode=="humain":
        plt.show()
    else:
        plt.close()




def analyse(path_name=None):
    if path_name is not None:
        directory = 'results/{0}/params.csv'.format(path_name)
    else:
        directory = min(glob.glob(os.path.join("results", '*/')), key=os.path.getmtime)
        directory = os.path.join(directory, "params.csv")
    
    # coefficient R² et reward selon les étapes d'entraînement
    df = pd.read_csv(directory)
    
    print(df)
    
    plt.figure(figsize=(8, 4))
    
    plt.xlabel("step")
    plt.grid(True, linestyle='--')
    plt.scatter(df["current_step"], df["r2_coef"], label="R² coef")
    plt.scatter(df["current_step"], df["reward"], label="reward")
    plt.legend()
    
    plt.figure(figsize=(2*4, 4))
    
    plt.subplot(1, 2, 1)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.grid(True, linestyle='--')
    plt.scatter(df["Re(a)"], df["Im(a)"])
    
    plt.subplot(1, 2, 2)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.grid(True, linestyle='--')
    plt.bar(df["current_step"], df["done"], width=1.)
    
    plt.tight_layout()
    plt.show()