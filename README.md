![Python](https://img.shields.io/badge/python-3.6-blue.svg)


# Search-Parameters
Travail sur la recherche optimale de param�tres d'inconnus.




## Installation

N�cessite `Python` *version 3.6*, ainsi que `tensorflow-gpu` *version 1.15* ou 
`tensorflow` *version 1.15*.

Attention, la version GPU n'est � utiliser que si vous disposez d'un GPU Nvidia
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

Pour acc�der au nombre de mode : `configs.environment.Config.N_MODES`

Pour acc�der au nombre d'actions possibles pour que le mod�le trouve les 
param�tres inconnus : `configs.environment.Config.MAX_STEPS`

Pour acc�der au nombre d'it�rations lors de l'entra�nement du mod�le : 
`configs.dqn.config.nsteps_train`





## Suivis

Pour visualiser l'avancement du calcul en temps r�el, il suffit de lancer 
`tensorboard`.

Il faut executer :
```shell
tensorboard --logdir results
```

Puis aller sur ![http://localhost:6006/](http://localhost:6006/) (le port 6006
�tant celui utilis� par `tensorboard`).