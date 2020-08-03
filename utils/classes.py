# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:50:42 2020

@author: alspe
"""


import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from configs.environment import Config



j = complex(0, 1)

class VFA_Ilyes:
        
    def __init__(self, n_modes, w_values, mat):
        self.n            = n_modes
        self.w_values     = w_values # 100000 valeurs
        self.mat          = mat
        self.sub_w_values = [w for i, w in enumerate(self.w_values) if i%100==0] # 1000 valeurs
        self.b_           = mat.rho_0*mat.k_0*mat.alpha_inf/(mat.eta*mat.phi)
        self.alpha_inf    = mat.alpha_inf
        
    
    def update(self, a=None, b=None, c=None, d=None, e=None, f=None, g=None, r_k=None, s_k=None, mu_k=None, xi_k=None):
        if a is not None:
            self.a    = a
        if b is not None:
            self.b    = b
        if c is not None:
            self.c    = c
        if d is not None:
            self.d    = d
        if e is not None:
            self.e    = e
        if f is not None:
            self.f    = f
        if g is not None:
            self.g    = g
        if r_k is not None:
            self.r_k  = r_k
        if s_k is not None:
            self.s_k  = s_k
        if mu_k is not None:
            self.mu_k = mu_k
        if xi_k is not None:
            self.xi_k = xi_k
        
    
    def evaluate(self, w):
        # s = jw avec w € R*+
        
        if Config.CASE_STUDY == "alpha":
            sum2 = 0
            for k in range(self.n):
                # Les xi doivent être positifs, donc on prend la valeur absolue
                sum2 += self.mu_k[k] / (j*w + abs(self.xi_k[k]))
            f = self.alpha_inf + self.alpha_inf/(self.b_*j*w) + sum2
            return f
        
        elif Config.CASE_STUDY == "f(w)":
            sum1, sum2 = 0, 0
            for k in range(self.n):
                sum1 += self.r_k[k]  / (j*w - self.s_k[k])
                sum2 += self.mu_k[k] / (j*w - self.xi_k[k])
            f = self.a*j*w + self.b + self.c/(j*w) + (self.d*j*w + self.e)*sum1 + (self.f*j*w + self.g)*sum2
            return f
    
            
    def compute(self, all_values=True):
        # On ne prend qu'un sous ensemble, car avec les 100000 de self.w_values,
        # cette fonction prend ~1.3s
        # Avec seulement 1000 la fonction dure maintenant ~0.016s
        if all_values:
            w_list = self.w_values
        else:
            w_list = self.sub_w_values
        
        f_values = []
        for w in w_list:
            f_values += [self.evaluate(w)]
        
        df = pd.DataFrame({"w": w_list,
                           "alpha": f_values,
                           "alpha_r": list(z.real for z in f_values),
                           "alpha_i": list(z.imag for z in f_values)})
        return df
    
            
    def save(self, path_name):
        # Enregistre les paramètres déterminés sous format JSON
        if not os.path.isdir('results'):
            os.mkdir('results')
            
        df = self.compute()
        df.to_csv("results/{0}/tortuosite.csv".format(path_name), index=False)




class Summary:
        
    def __init__(self):
        # On vérifie qu'il existe un dossier "results"
        if not os.path.isdir('results'):
            os.mkdir('results')
        
        # On crée le dossier de résultats du nom de la date
        date = datetime.datetime.today()
        self.path_name = date.isoformat().replace(":","-").split(".")[0]
        os.mkdir('results/{0}'.format(self.path_name))
        os.mkdir('results/{0}/fig'.format(self.path_name))
        
        self.file_name = 'results/{0}/params.csv'.format(self.path_name)
        
        
    def create(self):
        # On initialise le fichier csv par l'entête
        with open(self.file_name, "w") as file:
            file.write("current_step,r2_coef,reward,Re(mu[0]),Im(mu[0]),done,max_r2_coef\n")
            
        
    def update(self, current_step, r2_coef, reward, mu, done, max_r2_coef):
        # On ajoute une ligne au fichier
        with open(self.file_name, "a") as file:
            file.write("{0},{1},{2},{3},{4},{5},{6}\n".format(current_step, 
                                                      r2_coef, 
                                                      reward, 
                                                      mu.real, 
                                                      mu.imag,
                                                      done,
                                                      max_r2_coef))
            
    def plot(self, mode="humain"):    
        # On charge les résultats
        df = pd.read_csv(self.file_name)
        
        if Config.REWARD_TYPE=="re_coef":
            label        = "R² coef"
            label_extrem = "max(R²)"
        elif Config.REWARD_TYPE=="mse":
            label        = "MSE"
            label_extrem = "min(MSE)"
        
        # coefficient R² et reward selon les étapes d'entraînement
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        plt.xlabel("step")
        plt.grid(True, linestyle='--')
        plt.scatter(df["current_step"], df["r2_coef"], label=label)
        plt.scatter(df["current_step"], df["max_r2_coef"], label=label_extrem)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.xlabel("step")
        plt.grid(True, linestyle='--')
        plt.scatter(df["current_step"], df["reward"], label="reward")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/{0}/fig/reward.png'.format(self.path_name))
        if mode=="humain":
            plt.show()
        else:
            plt.close()
        
        # Affichage des valeurs du complexe "a"
        plt.figure(figsize=(4+1, 4))
        
        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.grid(True, linestyle='--')
        plt.scatter(df["Re(mu[0])"], df["Im(mu[0])"], c=df["current_step"], label=None)
        plt.colorbar(label='step')
        
        plt.tight_layout()
        plt.savefig('results/{0}/fig/param_mu_0.png'.format(self.path_name))
        if mode=="humain":
            plt.show()
        else:
            plt.close()
        
        # Affichage des étape "terminées"
        plt.figure(figsize=(4, 4))
        
        plt.yticks([0, 1], ("continue", "done"))
        plt.bar(df["current_step"], df["done"], width=1.)
        
        plt.tight_layout()
        plt.savefig('results/{0}/fig/done.png'.format(self.path_name))
        if mode=="humain":
            plt.show()
        else:
            plt.close()