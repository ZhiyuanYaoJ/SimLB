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

user_conf[0] = {
    'METHODS' : {
        "rlb-sac": {# SAC model
            "config": {
                'SAC_training_confs_': {
                    'hidden_dim': 100,
                    'action_range': 1.,
                    'batch_size': 64,
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
