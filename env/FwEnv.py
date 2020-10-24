# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 19:52:16 2020

@author: alspe
"""


import json
import random
import numpy as np

from utils.functions     import correlation, plot, mse

from configs.environment import Config, Transform

from env.TortuositeEnv   import TortuositeEnv



class AlphaEnv(TortuositeEnv):
        
    def _get_space_shapes(self):
        shapes = [Transform.N_ACT_PARAMS*2*Config.N_PARAMS + 1,
                  (2, 4, Config.N_MODES+2)]
        return shapes
    

    def _next_observation(self):

        # [[[ Re(a)  Re(e)  Re(r_k[0])  Re(r_k[1])  Re(r_k[2])  ... Re(r_k[N_MODES])  ]
        #   [ Re(b)  Re(f)  Re(s_k[0])  Re(s_k[1])  Re(s_k[2])  ... Re(s_k[N_MODES])  ]
        #   [ Re(c)  Re(g)  Re(mu_k[0]) Re(mu_k[1]) Re(mu_k[2]) ... Re(mu_k[N_MODES]) ]
        #   [ Re(d)  0.     Re(xi_k[0]) Re(xi_k[1]) Re(xi_k[2]) ... Re(xi_k[N_MODES]) ]]
        
        #  [[ Im(a)  Im(e)  Im(r_k[0])  Im(r_k[1])  Im(r_k[2])  ... Im(xi_k[N_MODES]) ]
        #   [ Im(b)  Im(f)  Im(s_k[0])  Im(s_k[1])  Im(s_k[2])  ... Im(s_k[N_MODES])  ]
        #   [ Im(c)  Im(g)  Im(mu_k[0]) Im(mu_k[1]) Im(mu_k[2]) ... Im(mu_k[N_MODES]) ]
        #   [ Im(d)  0.     Im(xi_k[0]) Im(xi_k[1]) Im(xi_k[2]) ... Im(xi_k[N_MODES]) ]]]
        
        list_of_el = [np.append([self.b, self.f], self.s_k),
                      np.append([self.c, self.g], self.mu_k),
                      np.append([self.d, 0.    ], self.xi_k)]
        
        re, im = np.array([[np.append([self.a, self.e], self.r_k).real]]), np.array([[np.append([self.a, self.e], self.r_k).imag]])
        for el in list_of_el:
            re = np.append(re, [[el.real]], axis=1)
            im = np.append(im, [[el.imag]], axis=1)
            
        # On rassemble des deux "canaux"
        obs = np.append(re, im, axis=0)
        return obs
    

    def _take_action(self, action):
        # On récupère les informations de l'action
        fact = action//Transform.N_ACT_PARAMS
        rest = action%Transform.N_ACT_PARAMS
        transf = Transform.TRANSF
        
        # Param Re(a)
        if fact < (1):
            self.a += complex(transf[rest], 0)
            
        # Param Im(a)
        elif fact < (2):
            self.a += complex(0, transf[rest])
        
        # Param Re(b)
        elif fact < (3):
            self.b += complex(transf[rest], 0)
            
        # Param Im(b)
        elif fact < (4):
            self.b += complex(0, transf[rest])
        
        # Param Re(c)
        elif fact < (5):
            self.c += complex(transf[rest], 0)
            
        # Param Im(c)
        elif fact < (6):
            self.c += complex(0, transf[rest])
        
        # Param Re(d)
        elif fact < (7):
            self.d += complex(transf[rest], 0)
            
        # Param Im(d)
        elif fact < (8):
            self.d += complex(0, transf[rest])
        
        # Param Re(e)
        elif fact < (9):
            self.e += complex(transf[rest], 0)
            
        # Param Im(e)
        elif fact < (10):
            self.e += complex(0, transf[rest])
        
        # Param Re(f)
        elif fact < (11):
            self.f += complex(transf[rest], 0)
            
        # Param Im(f)
        elif fact < (12):
            self.f += complex(0, transf[rest])
        
        # Param Re(g)
        elif fact < (13):
            self.g += complex(transf[rest], 0)
            
        # Param Im(g)
        elif fact < (14):
            self.g += complex(0, transf[rest])
            
        # Param Re(r_k)
        elif fact < (2*(7+0*Config.N_MODES) + 1*Config.N_MODES):
            idx = fact-2*(7+0*Config.N_MODES)
            self.r_k[idx]  += complex(transf[rest], 0)
            
        # Param Im(r_k)
        elif fact < (2*(7+0*Config.N_MODES) + 2*Config.N_MODES):
            idx = fact-2*(7+0*Config.N_MODES)-1*Config.N_MODES
            self.r_k[idx]  += complex(0, transf[rest])
            
        # Param Re(s_k)
        elif fact < (2*(7+1*Config.N_MODES) + 1*Config.N_MODES):
            idx = fact-2*(7+1*Config.N_MODES)
            self.s_k[idx]  += complex(transf[rest], 0)
            
        # Param Im(s_k)
        elif fact < (2*(7+1*Config.N_MODES) + 2*Config.N_MODES):
            idx = fact-2*(7+1*Config.N_MODES)-1*Config.N_MODES
            self.s_k[idx]  += complex(0, transf[rest])
            
        # Param Re(mu_k)
        elif fact < (2*(7+2*Config.N_MODES) + 1*Config.N_MODES):
            idx = fact-2*(7+2*Config.N_MODES)
            self.mu_k[idx] += complex(transf[rest], 0)
            
        # Param Im(mu_k)
        elif fact < (2*(7+2*Config.N_MODES) + 2*Config.N_MODES):
            idx = fact-2*(7+2*Config.N_MODES)-1*Config.N_MODES
            self.mu_k[idx] += complex(0, transf[rest])
            
        # Param Re(xi_k)
        elif fact < (2*(7+3*Config.N_MODES) + 1*Config.N_MODES):
            idx = fact-2*(7+3*Config.N_MODES)
            self.xi_k[idx] += complex(transf[rest], 0)
            
        # Param Im(xi_k)
        elif fact < (2*(7+3*Config.N_MODES) + 2*Config.N_MODES):
            idx = fact-2*(7+3*Config.N_MODES)-1*Config.N_MODES
            self.xi_k[idx] += complex(0, transf[rest])
            
        # On ne fait rien
        #else:
            # Rien

        self.vfa.update(self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.r_k, self.s_k, self.mu_k, self.xi_k)
        pred_df = self.vfa.compute(all_values=False)

        if Config.REWARD_TYPE=="re_coef":
            self.r2_coef = correlation(self.sub_df, pred_df)
            if self.r2_coef > self.r2_coef_max:
                self.r2_coef_max = self.r2_coef
                
        elif Config.REWARD_TYPE=="mse":
            self.mse = mse(self.sub_df, pred_df)
            if self.mse < self.mse_min:
                self.mse_min = self.mse
                
        else:
            raise NotImplementedError


    def reset(self):
        # On initialise les paramètres de la fonction
        self.a    = Config.MAX_INITIALIZE*complex(random.random(), random.random())
        self.b    = Config.MAX_INITIALIZE*complex(random.random(), random.random())
        self.c    = Config.MAX_INITIALIZE*complex(random.random(), random.random())
        self.d    = Config.MAX_INITIALIZE*complex(random.random(), random.random())
        self.e    = Config.MAX_INITIALIZE*complex(random.random(), random.random())
        self.f    = Config.MAX_INITIALIZE*complex(random.random(), random.random())
        self.g    = Config.MAX_INITIALIZE*complex(random.random(), random.random())
        self.r_k  = Config.MAX_INITIALIZE*np.random.rand(Config.N_MODES, 2).view(dtype=np.complex128).flatten()
        self.s_k  = Config.MAX_INITIALIZE*np.random.rand(Config.N_MODES, 2).view(dtype=np.complex128).flatten()
        self.mu_k = Config.MAX_INITIALIZE*np.random.rand(Config.N_MODES, 2).view(dtype=np.complex128).flatten()
        self.xi_k = Config.MAX_INITIALIZE*np.random.rand(Config.N_MODES, 2).view(dtype=np.complex128).flatten()
        
        self.vfa.update(self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.r_k, self.s_k, self.mu_k, self.xi_k)
        pred_df        = self.vfa.compute(all_values=False)
        self.r2_coef_0 = correlation(self.sub_df, pred_df)
        self.mse_0     = mse(self.sub_df, pred_df)
        
        # Paramètres utils
        if Config.REWARD_TYPE=="re_coef":
            self.r2_coef     = self.r2_coef_0
            self.r2_coef_max = self.r2_coef_0
        elif Config.REWARD_TYPE=="mse":
            self.mse     = self.mse_0
            self.mse_min = self.mse_0
        else:
            raise NotImplementedError

        # Set the current step to a random point within the data frame
        self.current_step = 0

        return self._next_observation()
        

    def _save_params(self):
        # Enregistre les paramètres déterminés sous format JSON
            
        data = {
                "a":    {"Re": self.a.real, "Im": self.a.imag},
                "b":    {"Re": self.b.real, "Im": self.b.imag},
                "c":    {"Re": self.c.real, "Im": self.c.imag},
                "d":    {"Re": self.d.real, "Im": self.d.imag},
                "e":    {"Re": self.e.real, "Im": self.e.imag},
                "f":    {"Re": self.f.real, "Im": self.f.imag},
                "g":    {"Re": self.g.real, "Im": self.g.imag},
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
        pred_df              = self.vfa.compute(all_values=False)
        r2_coef_r, r2_coef_i = correlation(self.sub_df, pred_df, split=True)
        mse_r, mse_i         = mse(self.sub_df, pred_df, split=True)
        
        print()
        print("#"*80)
        print("Étape : {0}".format(self.current_step))
        print("R² coef :     {0:+.6f}".format((r2_coef_r + r2_coef_i)/2.))
        print("  ~ initial : {0:+.6f}".format(self.r2_coef_0))
        if Config.REWARD_TYPE=="re_coef":
            print("  ~ max :     {0:+.6f}".format(self.r2_coef_max))
        print("MSE valeur :  {0:+.6f}".format((mse_r + mse_i)/2.))
        print("  ~ initial : {0:+.6f}".format(self.mse_0))
        if Config.REWARD_TYPE=="mse":
            print("  ~ min :     {0:+.6f}".format(self.mse_min))
        print("Détails :")
        print("   R² ~ partie réelle :      {0:+.6f}".format(r2_coef_r))
        print("   R² ~ partie imaginaire :  {0:+.6f}".format(r2_coef_i))
        print("   MSE ~ partie réelle :     {0:+.6f}".format(mse_r))
        print("   MSE ~ partie imaginaire : {0:+.6f}".format(mse_i))
        
        print("-"*80)
        print("Les paramètres trouvés :")
        print("~ "*40)
        print("   a     = {0:+.6e} + {1:+.6e}j".format(self.a.real, self.a.imag))
        print("   b     = {0:+.6e} + {1:+.6e}j".format(self.b.real, self.b.imag))
        print("   c     = {0:+.6e} + {1:+.6e}j".format(self.c.real, self.c.imag))
        print("   d     = {0:+.6e} + {1:+.6e}j".format(self.d.real, self.d.imag))
        print("   e     = {0:+.6e} + {1:+.6e}j".format(self.d.real, self.d.imag))
        print("   f     = {0:+.6e} + {1:+.6e}j".format(self.d.real, self.d.imag))
        print("   g     = {0:+.6e} + {1:+.6e}j".format(self.d.real, self.d.imag))
        print("~ "*40)
        for k in range(Config.N_MODES):
            print("   r[{0:1d}]  = {1:+.6e} + {2:+.6e}j".format(k+1, self.r_k[k].real, self.r_k[k].imag))
        print("~ "*40)
        for k in range(Config.N_MODES):
            print("   s[{0:1d}]  = {1:+.6e} + {2:+.6}j".format(k+1, self.s_k[k].real, self.s_k[k].imag))
        print("~ "*40)
        for k in range(Config.N_MODES):
            print("   mu[{0:1d}] = {1:+.6e} + {2:+.6e}j".format(k+1, self.mu_k[k].real, self.mu_k[k].imag))
        print("~ "*40)
        for k in range(Config.N_MODES):
            print("   xi[{0:1d}] = {1:+.6e} + {2:+.6e}j".format(k+1, self.xi_k[k].real, self.xi_k[k].imag))
    
        print("~ "*40)
        print("Pour la fonction :")
        print()
        
        screen  = ["                   c  ",
                   "f(s) =  a*s + b + --- ",
                   "                   s  "]
        for k in range(3):
            print(screen[k])
        print()
            
        screen = ["                     / ",
                  "      + (d*s + e) * |  ",
                  "                     \\ "]
        for k in range(Config.N_MODES+1):
            screen = self.print_fraction(screen, k, 1)
        for k in range(3):
            print(screen[k])    
        print()
        
        screen = ["                     / ",
                  "      + (f*s + g) * |  ",
                  "                     \\ "]
        for k in range(Config.N_MODES+1):
            screen = self.print_fraction(screen, k, 2)
        for k in range(3):
            print(screen[k])
        print()
        
        if self.summary is not None:
            print("-"*80)
            self._save_params()
        
            print("-"*80)
            plot(self.df, self.vfa.compute(), mode=mode, path_name=self.summary.path_name)
            
            if mode=="humain":
                print("-"*80)
            self.summary.plot(mode=mode)
            
        print("#"*80)
        print()
    
    