# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:56:15 2020

@author: alspe
"""

import pandas as pd

from env.TortuositeEnv import TortuositeEnv

from core.q_schedule import LinearExploration, LinearSchedule
from core.q_linear   import Linear

from configs.dqn import config


# load data
df = pd.read_csv('data/tortuosite.csv')


    
# make env
env = TortuositeEnv(df)

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
    
