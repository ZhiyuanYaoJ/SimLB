# ---------------------------------------------------------------------------- #
#                                  Description                                 #
# This file stores user preferred configurations, which will overwrite default #
# global configuration (in global_conf.py)                                     #
# ---------------------------------------------------------------------------- #

from config.global_conf import *

# ---------------------------------------------------------------------------- #
#                                      Dev                                     #
# ---------------------------------------------------------------------------- #

DEBUG = 0
user_conf = {}

'''
u = {
    'METHODS' : {
        "rlb-sac": {# SAC model
            "config": {
                'SAC_training_confs_': {
                    'hidden_dim': 100,
                    'action_range': 1.,
                    'batch_size': 65,
                    'update_itr': 10,
                    'reward_scale': 10.,
                    'save_interval': 100,  # time interval for saving models, in seconds
                    'AUTO_ENTROPY': True,
                    'model_path': 'sac_v2',
                }
            }
        }
    }
}
#hidden = [5, 10, 50, 100, 200, 400, 1000]
#user_conf = {i:u for i in range (len(hidden))}
#for i in range (len(hidden)):
    #u['METHODS']['rlb-sac']['config']['SAC_training_confs_']['hidden_dim'] = hidden[i]
'''