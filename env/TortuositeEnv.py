# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 00:44:47 2020

@author: alspe
"""


import json
import random
import numpy as np

from utils.functions     import correlation, plot
from utils.classes       import VFA_Ilyes, Summary

from configs.environment import Config, Transform


class Discrete(object):
    def __init__(self, n):
        self.shape = n

    def sample(self):
        return np.random.randint(0, self.shape)


class Box(object):
    def __init__(self, shape, dtype=np.float16):
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.rand(self.shape[0], self.shape[1], self.shape[2]).astype(self.dtype)
        

class TortuositeEnv(object):#gym.Env):
    """A fonction approximation environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(TortuositeEnv, self).__init__()

        self.df     = df
        self.sub_df = df.drop(df[df.index%100!=0].index)
        self.vfa    = VFA_Ilyes(Config.N_MODES, self.df["w"].values)
        
        # On crée le suivi de certaines valeurs
        self.summary = Summary()
        self.summary.create()

        # Actions
        # Transform.TRANSF * [Re Im] * N_PARAMS
        self.action_space = Discrete(Transform.N_ACTIONS*2*Config.N_PARAMS)        

        # Contient les valeurs de paramètres des cinq précédentes estimations
        self.observation_space = Box(shape=(Config.N_MODES, Config.N_INCONNUES, 2), dtype=np.float16)


    def _next_observation(self):

        # [[[ Re(a)       0.          0.          ... 0.                ]
        #   [ Re(r_k[0])  Re(r_k[1])  Re(r_k[2])  ... Im(xi_k[N_MODES]) ]
        #   [ ...                                 ... ...               ]
        #   [ Re(xi_k[0]) Re(xi_k[1]) Re(xi_k[2]) ... Im(xi_k[N_MODES]) ]]
        
        #  [[ Im(a)       0.          0.          ... 0.                ]
        #   [ Im(r_k[0])  Im(r_k[1])  Im(r_k[2])  ... Im(xi_k[N_MODES]) ]
        #   [ ...                                 ... ...               ]
        #   [ Im(xi_k[0]) Im(xi_k[1]) Im(xi_k[2]) ... Im(xi_k[N_MODES]) ]]]
        
        re = np.array([[[self.a.real] + [0. for k in range(Config.N_MODES-1)]]])
        for el in [self.r_k, self.s_k, self.mu_k, self.xi_k]:
            re = np.append(re, [[el.real]], axis=1)

        im = np.array([[[self.a.imag] + [0. for k in range(Config.N_MODES-1)]]])
        for el in [self.r_k, self.s_k, self.mu_k, self.xi_k]:
            im = np.append(im, [[el.imag]], axis=1)
            
        # On rassemble des deux "canaux"
        obs = np.append(re, im, axis=0)
        return obs
    

    def _take_action(self, action):
        # On récupère les informations de l'action
        fact = action//Transform.N_ACTIONS
        rest = action%Transform.N_ACTIONS
        transf = Transform.TRANSF
        
        # Param Re(a)
        if fact < (1):
            self.a = self.a + complex(transf[rest], 0)
            
        # Param Im(a)
        elif fact < (2):
            self.a = self.a + complex(0, transf[rest])
            
        # Param Re(r_k)
        elif fact < (2*(1+0*Config.N_MODES) + 1*Config.N_MODES):
            idx = fact-2*(1+0*Config.N_MODES)
            self.r_k[idx]  = self.r_k[idx]  + complex(transf[rest], 0)
            
        # Param Im(r_k)
        elif fact < (2*(1+0*Config.N_MODES) + 2*Config.N_MODES):
            idx = fact-2*(1+0*Config.N_MODES)-1*Config.N_MODES
            self.r_k[idx]  = self.r_k[idx]  + complex(0, transf[rest])
            
        # Param Re(s_k)
        elif fact < (2*(1+1*Config.N_MODES) + 1*Config.N_MODES):
            idx = fact-2*(1+1*Config.N_MODES)
            self.s_k[idx]  = self.s_k[idx]  + complex(transf[rest], 0)
            
        # Param Im(s_k)
        elif fact < (2*(1+1*Config.N_MODES) + 2*Config.N_MODES):
            idx = fact-2*(1+1*Config.N_MODES)-1*Config.N_MODES
            self.s_k[idx]  = self.s_k[idx]  + complex(0, transf[rest])
            
        # Param Re(mu_k)
        elif fact < (2*(1+2*Config.N_MODES) + 1*Config.N_MODES):
            idx = fact-2*(1+2*Config.N_MODES)
            self.mu_k[idx] = self.mu_k[idx] + complex(transf[rest], 0)
            
        # Param Im(mu_k)
        elif fact < (2*(1+2*Config.N_MODES) + 2*Config.N_MODES):
            idx = fact-2*(1+2*Config.N_MODES)-1*Config.N_MODES
            self.mu_k[idx] = self.mu_k[idx] + complex(0, transf[rest])
            
        # Param Re(xi_k)
        elif fact < (2*(1+3*Config.N_MODES) + 1*Config.N_MODES):
            idx = fact-2*(1+3*Config.N_MODES)
            self.xi_k[idx] = self.xi_k[idx] + complex(transf[rest], 0)
            
        # Param Im(xi_k)
        elif fact < (2*(1+3*Config.N_MODES) + 2*Config.N_MODES):
            idx = fact-2*(1+3*Config.N_MODES)-1*Config.N_MODES
            self.xi_k[idx] = self.xi_k[idx] + complex(0, transf[rest])


        self.vfa.update(self.a, self.r_k, self.s_k, self.mu_k, self.xi_k)
        pred_df = self.vfa.compute(all_values=False)
        self.r2_coef = correlation(self.sub_df, pred_df)

        if self.r2_coef > self.max_r2_coef:
            self.max_r2_coef = self.r2_coef


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

#        done = (self.win) or (self.current_step == Config.MAX_STEPS)
        done = (self.r2_coef >= Config.MAX_R2) or (self.current_step == Config.MAX_STEPS)
        
        # Fonction de reward
        reward = self.r2_coef/Config.MAX_STEPS

        obs = self._next_observation()
        
        # On met à jour le suivi
        self.summary.update(self.current_step, self.r2_coef, reward, self.a, done, self.max_r2_coef)

        return obs, reward, done, {}


    def reset(self):
        # On initialise les paramètres de la fonction
        self.a    = Config.N_MODES*complex(random.random(), random.random())
        self.r_k  = Config.MAX_INITIALIZE*np.random.rand(Config.N_MODES, 2).view(dtype=np.complex128).flatten()
        self.s_k  = Config.MAX_INITIALIZE*np.random.rand(Config.N_MODES, 2).view(dtype=np.complex128).flatten()
        self.mu_k = Config.MAX_INITIALIZE*np.random.rand(Config.N_MODES, 2).view(dtype=np.complex128).flatten()
        self.xi_k = Config.MAX_INITIALIZE*np.random.rand(Config.N_MODES, 2).view(dtype=np.complex128).flatten()
        self.vfa.update(self.a, self.r_k, self.s_k, self.mu_k, self.xi_k)
        
        # Paramètres utils
        self.r2_coef = -1.
        self.max_r2_coef = -1.

        # Set the current step to a random point within the data frame
        self.current_step = 0

        return self._next_observation()
        

    def _save_params(self):
        # Enregistre les paramètres déterminés sous format JSON
            
        data = {"a":    {"Re": self.a.real, "Im": self.a.imag},
                "r_k":  {"r_{0}".format(i):  {"Re": r.real, "Im": r.imag}   for i, r  in enumerate(self.r_k)},
                "s_k":  {"s_{0}".format(i):  {"Re": s.real, "Im": s.imag}   for i, s  in enumerate(self.s_k)},
                "mu_k": {"mu_{0}".format(i): {"Re": mu.real, "Im": mu.imag} for i, mu in enumerate(self.mu_k)},
                "xi_k": {"xi_{0}".format(i): {"Re": xi.real, "Im": xi.imag} for i, xi in enumerate(self.xi_k)}
               }
        
        with open('results/{0}/params.json'.format(self.summary.path_name), 'w') as outfile:
            json.dump(data, outfile)
        print("Entregistrement des paramètres")
        
        self.vfa.save(self.summary.path_name)
        print("Entregistrement des données calculées")


    def render(self, mode='human', close=False):
        # Affichage des résultats
        
        print()
        print("###########################################")
        print("Étape : {0}".format(self.current_step))
        print("R² corrélation : {0:+.6f}".format(self.r2_coef))
        print("R² max trouvé :  {0:+.6f}".format(self.max_r2_coef))
        print("Détails :")
        pred_df = self.vfa.compute(all_values=False)
        r2coef_r, r2coef_i = correlation(self.sub_df, pred_df, split=True)
        print("   R² ~ partie réelle :    {0:+.6f}".format(r2coef_r))
        print("   R² ~ partie imaginaire : {0:+.6f}".format(r2coef_i))
        
        print("-------------------------------------------")
        print("Les paramètres trouvés :")
        print("   a     = {0:+.6e} + {1:+.6e}j".format(self.a.real, self.a.imag))
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
        for k in range(Config.N_MODES):
            print("   r[{0:1d}]  = {1:+.6e} + {2:+.6e}j".format(k, self.r_k[k].real, self.r_k[k].imag))
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
        for k in range(Config.N_MODES):
            print("   s[{0:1d}]  = {1:+.6e} + {2:+.6}j".format(k, self.s_k[k].real, self.s_k[k].imag))
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
        for k in range(Config.N_MODES):
            print("   mu[{0:1d}] = {1:+.6e} + {2:+.6e}j".format(k, self.mu_k[k].real, self.mu_k[k].imag))
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
        for k in range(Config.N_MODES):
            print("   xi[{0:1d}] = {1:+.6e} + {2:+.6e}j".format(k, self.xi_k[k].real, self.xi_k[k].imag))
    
        print("-------------------------------------------")
        self._save_params()
        
        print("-------------------------------------------")
        plot(self.df, self.vfa.compute(), mode=mode, path_name=self.summary.path_name)
        if mode=="humain":
            print("-------------------------------------------")
        self.summary.plot(mode=mode)
            
        print("###########################################")
        print()