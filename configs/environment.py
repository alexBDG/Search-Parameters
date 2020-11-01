# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 09:14:48 2020

@author: alspe
"""


import numpy as np



class Transform:
    EXPO_MAX     = 6
    TRANSF       = np.append(-pow(10, np.linspace(-EXPO_MAX, EXPO_MAX, 2*EXPO_MAX+1)), 
                       pow(10, np.linspace(-EXPO_MAX, EXPO_MAX, 2*EXPO_MAX+1)))
    N_ACT_PARAMS = len(TRANSF)
    
    
class Config:
    # "alpha" or "f(s)"
    CASE_STUDY = "alpha"
    
    # "r2_coef" or "mse"
    REWARD_TYPE = "mse" 
    
    IS_TENSORBOARD = True
    TRAIN          = True
    
    TRAIN_TIME_STEPS = 2000 # Inutile
    TEST_TIME_STEPS  = 100 # Inutile
    MAX_STEPS        = 1000
    MAX_INITIALIZE   = 1000000
    
    # Critères d'arrêt
    MAX_R2  = 0.95
    MIN_MSE = 0.5
    
    N_MODES     = 6 # k € [1, N_MODES]
    if CASE_STUDY == "alpha":
        N_INCONNUES = 2
        N_PARAMS    = 2*N_MODES # (mu_k, xi_k)*k
        N_ACTIONS   = Transform.N_ACT_PARAMS*N_PARAMS + 1
    elif CASE_STUDY == "f(s)":
        N_INCONNUES = 7 + 4
        N_PARAMS    = 7 + 4*N_MODES # a + b + c + d + e + f + g + (r_k, s_k, mu_k, xi_k)*k
        N_ACTIONS   = Transform.N_ACT_PARAMS*2*N_PARAMS + 1
    
    DIM_DENSE_1 = 4*N_ACTIONS
    DIM_DENSE_2 = 4*N_ACTIONS
       

class Materiau:
    rho_0     = 1000.
    k_0       = 1.
    alpha_0   = 0.1
    alpha_inf = 10.
    phi       = 0.5
    Delta     = 1.
    eta       = 1.84*10**(-5)
    
    
    
class IlyMou:
    rho_0     = 1.177         # kg/m^3
    k_0       = 1.63*10**(-9) # m^2
    alpha_0   = 2.3
    alpha_inf = 1.46
    phi       = 0.94
    Delta     = 91*10**(-6)   # m
    eta       = 1.84*10**(-5) # N.s.m−2