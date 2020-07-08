# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:50:42 2020

@author: alspe
"""


import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt




class VFA_Ilyes:
        
    def __init__(self, n_modes, w_values):
        self.n            = n_modes
        self.w_values     = w_values # 100000 valeurs
        self.sub_w_values = [w for i, w in enumerate(self.w_values) if i%100==0] # 1000 valeurs
        
    
    def update(self, a, r_k, s_k, mu_k, xi_k):
        self.a    = a
        self.r_k  = r_k
        self.s_k  = s_k
        self.mu_k = mu_k
        self.xi_k = xi_k
        
    
    def evaluate(self, s):
        sum1, sum2 = 0, 0
        for k in range(self.n):
            sum1 += self.r_k[k]  / (s - self.s_k[k])
            sum2 += self.mu_k[k] / (s - self.xi_k[k])
        f = s + self.a + (sum1 + sum2)*s
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
            file.write("current_step,r2_coef,reward,Re(a),Im(a),done,max_r2_coef\n")
            
        
    def update(self, current_step, r2_coef, reward, a, done, max_r2_coef):
        # On ajoute une ligne au fichier
        with open(self.file_name, "a") as file:
            file.write("{0},{1},{2},{3},{4},{5},{6}\n".format(current_step, 
                                                      r2_coef, 
                                                      reward, 
                                                      a.real, 
                                                      a.imag,
                                                      done,
                                                      max_r2_coef))
            
    def plot(self, mode="humain"):    
        # On charge les résultats
        df = pd.read_csv(self.file_name)
        
        # coefficient R² et reward selon les étapes d'entraînement
        plt.figure(figsize=(8, 4))
        
        plt.xlabel("step")
        plt.grid(True, linestyle='--')
        plt.scatter(df["current_step"], df["r2_coef"], label="R² coef")
        plt.scatter(df["current_step"], df["max_r2_coef"], label="max(R²)")
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
        plt.axis(aspect='equal')
        plt.grid(True, linestyle='--')
        plt.scatter(df["Re(a)"], df["Im(a)"], c=df["current_step"], label=None)
        plt.colorbar(label='step')
        
        plt.tight_layout()
        plt.savefig('results/{0}/fig/param_a.png'.format(self.path_name))
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