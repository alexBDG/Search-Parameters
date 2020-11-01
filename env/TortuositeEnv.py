# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 00:44:47 2020

@author: alspe
"""


import numpy as np

from utils.classes       import VFA_Ilyes, Summary

from configs.environment import Config


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
        

class TortuositeEnv(object):
    """A fonction approximation environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, mat, summary=False):
        super(TortuositeEnv, self).__init__()

        self.df     = df
        self.mat    = mat
        self.sub_df = df.drop(df[df.index%1000!=0].index)
        self.vfa    = VFA_Ilyes(Config.N_MODES, self.df["w"].values, self.mat,
                                df.alpha_r.values, df.alpha_i.values)
        
        # On crée le suivi de certaines valeurs
        if summary:
            self.summary = Summary()
            self.summary.create()
        else:
            self.summary = None
        
        shapes = self._get_space_shapes()
        # Actions
        # Transform.TRANSF * [Re Im] * N_PARAMS + 1
        # L'action supplémentaire est pour le cas où l'on ne fait rien
        self.action_space = Discrete(shapes[0])        

        # Contient les valeurs de paramètres des cinq précédentes estimations
        self.observation_space = Box(shape=shapes[1], dtype=np.float16)
        
    
    def _get_space_shapes(self):
        raise NotImplementedError
        
    
    def _next_observation(self):
        raise NotImplementedError
    

    def _take_action(self, action):
        raise NotImplementedError


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

#        done = (self.win) or (self.current_step == Config.MAX_STEPS)
        if Config.REWARD_TYPE=="re_coef":
            done = (self.r2_coef >= Config.MAX_R2) or (self.current_step == Config.MAX_STEPS)
        elif Config.REWARD_TYPE=="mse":
            done = (self.mse <= Config.MIN_MSE) or (self.current_step == Config.MAX_STEPS)
        
        # Fonction de reward        
        if Config.REWARD_TYPE=="re_coef":
            reward = self.r2_coef/Config.MAX_STEPS
        elif Config.REWARD_TYPE=="mse":
            reward = 1./(1. + self.mse)/Config.MAX_STEPS
        if done and self.current_step < Config.MAX_STEPS:
            # Si done==True après Config.MAX_STEPS/2 étapes, alors reward=5
            reward = 10*(Config.MAX_STEPS-self.current_step+1)/Config.MAX_STEPS

        obs = self._next_observation()
        
        # On met à jour le suivi
        if Config.REWARD_TYPE=="re_coef":
            value        = self.r2_coef
            value_extrem = self.max_r2_coef
        elif Config.REWARD_TYPE=="mse":
            value        = self.mse
            value_extrem = self.mse_min
            
        if self.summary is not None:
            self.summary.update(self.current_step, value, reward, self.mu_k[0], done, value_extrem)

        return obs, reward, done, {}


    def reset(self):
        raise NotImplementedError
        

    def _save_params(self):
        raise NotImplementedError
        
        
    def _print_fraction(self, screen, k, n_fract):
        if k==Config.N_MODES:
            screen[0] += " \\"
            screen[1] += "  |"
            screen[2] += " /"
            return screen
        elif k>0:
            screen[0] += "   "
            screen[1] += " + "
            screen[2] += "   "
        
        if n_fract==1:
            screen[0] += "   r[{0}]   ".format(k+1)
            screen[1] += "----------"
            screen[2] += "s[{0}] + j*w".format(k+1)
        elif n_fract==2:
            screen[0] += "   mu[{0}]   ".format(k+1)
            screen[1] += "-----------"
            screen[2] += "xi[{0}] + j*w".format(k+1)
        
        return screen
 

    def render(self, mode='human', close=False):
        raise NotImplementedError