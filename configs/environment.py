# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 09:14:48 2020

@author: alspe
"""


import numpy as np


class Config:
    IS_TENSORBOARD = True
    TRAIN          = True
    
    TRAIN_TIME_STEPS = 2000
    TEST_TIME_STEPS  = 100
    MAX_STEPS        = 1000
    MAX_R2           = 1.
    MAX_INITIALIZE   = 1000000
    
    N_MODES     = 2 # k € [1, N_MODES]
    N_INCONNUES = 1 + 4
    N_PARAMS    = 1 + 4*N_MODES # a + (r_k, s_k, mu_k, xi_k)*k
    N_ACTIONS   = 8*2*N_PARAMS
    
    DIM_DENSE_1 = 4*N_ACTIONS
    DIM_DENSE_2 = 4*N_ACTIONS
    
    

class Transform:
    EXPO_MAX  = 3
    TRANSF    = np.append(-pow(10, np.linspace(-EXPO_MAX, EXPO_MAX, 2*EXPO_MAX+1)), 
                       pow(10, np.linspace(-EXPO_MAX, EXPO_MAX, 2*EXPO_MAX+1)))
    N_ACTIONS = len(TRANSF)

    

class Materiau:
    rho_0 = 1000.
    k_0 = 1.
    alpha_0 = 0.1
    alpha_inf = 10.
    phi = 0.5
    Delta = 1.
    eta = 1.84*10**(-5)
    
    
    
class IlyMou:
    rho_0 = 1.177 # kg/m^3
    k_0 = 1.63*10**(-9) # m^2
    alpha_0 = 2.3
    alpha_inf = 1.46
    phi = 0.94
    Delta = 91*10**(-6) # m
    eta = 1.84*10**(-5) # N.s.m−2