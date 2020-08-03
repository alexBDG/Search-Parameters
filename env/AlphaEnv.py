# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 19:52:16 2020

@author: alspe
"""


import json
import numpy as np

from utils.functions     import correlation, plot, mse

from configs.environment import Config, Transform

from env.TortuositeEnv   import TortuositeEnv



class AlphaEnv(TortuositeEnv):
        
    def _get_space_shapes(self):
        shapes = [Transform.N_ACT_PARAMS*Config.N_PARAMS + 1,
                  (1, 2, Config.N_MODES)]
        return shapes
        

    def _next_observation(self):

        # [[[ mu_k[0] mu_k[1] mu_k[2] ... mu_k[N_MODES] ]
        #   [ xi_k[0] xi_k[1] xi_k[2] ... xi_k[N_MODES] ]]]
            
        # On rassemble des deux "canaux"
        obs = np.append([[self.mu_k]], [[self.xi_k]], axis=1)
        return obs
    

    def _take_action(self, action):
        # On récupère les informations de l'action
        fact = action//Transform.N_ACT_PARAMS
        rest = action%Transform.N_ACT_PARAMS
        transf = Transform.TRANSF
            
        # Param mu_k
        if fact < 1*Config.N_MODES:
            idx = fact-0*Config.N_MODES
            self.mu_k[idx] += transf[rest]
            
        # Param xi_k
        elif fact < 2*Config.N_MODES:
            idx = fact-1*Config.N_MODES
            self.xi_k[idx] += transf[rest]
            
        # On ne fait rien
        #else:
            # Rien

        self.vfa.update(mu_k=self.mu_k, xi_k=self.xi_k)
        pred_df      = self.vfa.compute(all_values=False)

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
        self.mu_k = Config.MAX_INITIALIZE*np.random.rand(Config.N_MODES)
        self.xi_k = Config.MAX_INITIALIZE*np.random.rand(Config.N_MODES)
        
        self.vfa.update(mu_k=self.mu_k, xi_k=self.xi_k)
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
        print("###########################################")
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
        
        print("-------------------------------------------")
        print("Les paramètres trouvés :")
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
        for k in range(Config.N_MODES):
            print("   mu[{0:1d}] = {1:+.6e}".format(k+1, self.mu_k[k]))
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
        for k in range(Config.N_MODES):
            print("   xi[{0:1d}] = {1:+.6e}".format(k+1, abs(self.xi_k[k])))
    
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
        print("Pour la fonction :")
        print()
        
        screen = ["                        alpha_inf    1  ",
                  "alpha(w) =  alpha_inf + --------- * --- ",
                  "                            b       j*w "]
        for k in range(3):
            print(screen[k])
        print()
        
        screen = ["               / ",
                  "           +  |  ",
                  "               \\ "]
        for k in range(Config.N_MODES+1):
            screen = self._print_fraction(screen, k, 2)
        for k in range(3):
            print(screen[k])
        print()
        
        if self.summary is not None:
            print("-------------------------------------------")
            self._save_params()
        
            print("-------------------------------------------")
            plot(self.df, self.vfa.compute(), mode=mode, path_name=self.summary.path_name)
            
            if mode=="humain":
                print("-------------------------------------------")
            self.summary.plot(mode=mode)
            
        print("###########################################")
        print()
    
    