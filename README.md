![Python](https://img.shields.io/badge/python-3.6-blue.svg)


# Search-Parameters
Travail sur la recherche optimale de paramêtres d'inconnus.




## Installation

Nécessite `Python` *version 3.6*, ainsi que `tensorflow-gpu` *version 1.15* ou 
`tensorflow` *version 1.15*.

Attention, la version GPU n'est à utiliser que si vous disposez d'un GPU Nvidia
 compatible CUDA.

Un fichier *requirements.txt* est fourni pour l'installation. Si vous ne 
souhaitez pas installer la version GPU de TensorFlow, remplacez la 
**ligne 29** :
```
tensorflow-gpu==1.15.0
```
Par :
```
tensorflow==1.15.0
```


Son utilisation est la suivante :
```shell
pip install -r requirements.txt
```




## Explications

Pour accéder au nombre de mode : `configs.environment.Config.N_MODES`

Pour accéder au nombre d'actions possibles pour que le modèle trouve les 
paramètres inconnus : `configs.environment.Config.MAX_STEPS`

Pour accéder au nombre d'itérations lors de l'entraînement du modèle : 
`configs.dqn.config.nsteps_train`





## Suivis

Pour visualiser l'avancement du calcul en temps réel, il suffit de lancer 
`tensorboard`.

Il faut executer :
```shell
tensorboard --logdir results
```

Puis aller sur ![http://localhost:6006/](http://localhost:6006/) (le port 6006
étant celui utilisé par `tensorboard`).