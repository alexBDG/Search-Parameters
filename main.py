# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:56:15 2020

@author: alspe
"""

import pandas as pd

from env.AlphaEnv import AlphaEnv
#from env.FwEnv import FwEnv

from core.q_schedule import LinearExploration, LinearSchedule
from core.q_linear   import Linear

from configs.dqn         import config
#from configs.linear      import config
from configs.environment import IlyMou, Config


# load data
df = pd.read_csv('data/tortuosite.csv')

# load material
mat = IlyMou()


# make env
if Config.CASE_STUDY=="alpha":
    env = AlphaEnv(df, mat)
elif Config.CASE_STUDY=="f(w)":
#    env = FwEnv(df, mat)
    raise NotImplementedError
else:
    raise NotImplementedError

# exploration strategy
exp_schedule = LinearExploration(env,
                                 config.eps_begin,
                                 config.eps_end,
                                 config.eps_nsteps)

# learning rate schedule
lr_schedule  = LinearSchedule(config.lr_begin,
                              config.lr_end,
                              config.lr_nsteps)

# train model
model = Linear(env, config)
model.run(exp_schedule, lr_schedule)
    
