# ---------------------------------------------------------------------------- #
#                                  Description                                 #
# This file stores a simple tier4 confiuration, which will overwrite default #
# global configuration (in global_conf.py)                                     #
# ---------------------------------------------------------------------------- #

from config.global_conf import *
from config.node_register import METHODS, NODE_MAP

# ---------------------------------------------------------------------------- #
#                                      Log                                     #
# ---------------------------------------------------------------------------- #

#LOG_FOLDER = '../data/simulation' # overwrite

# ---------------------------------------------------------------------------- #
#                                   Topology                                   #
# ---------------------------------------------------------------------------- #

def generate_node_config_tier4(
    lb_method='ecmp',
    n_clt=1, 
    n_er=1, 
    n_lb=N_LB, 
    n_as=N_AS, 
    n_worker_baseline=N_WORKER_BASELINE, 
    n_worker2change=N_WORKER2CHANGE, 
    n_worker_multiplier=N_WORKER_MULTIPLIER,
    n_worker_multiplier_distribution=N_WORKER_MULTIPLIER_DISTRIBUTION,
    as_mp_level=AS_MULTIPROCESS_LEVEL,
    lb_bucket_size=LB_BUCKET_SIZE,
    log_folder=LOG_FOLDER,
    rl_test=False,
    debug=DEBUG):
    clt_ids = list(range(n_clt))
    er_ids = list(range(n_er))
    lb_ids = list(range(n_lb))
    as_ids = list(range(n_as))

    clt_template = {
        'child_ids': er_ids,
        'child_prefix': 'er',  # connected to edge router
        'debug': 0
    }

    er_template = {
        'child_ids': lb_ids,
        'child_prefix': 'lb',  # connected to load balancers
    }

    lb_template = {
        'child_ids': as_ids,
        'debug': 0,
        'bucket_size': lb_bucket_size,
    }

    as_template = {
        'n_worker': n_worker_baseline,
        'multiprocess_level': as_mp_level,
        'debug': 0
    }

    clt_config = {i: clt_template.copy() for i in clt_ids}
    er_config = {i: er_template.copy() for i in er_ids}
    as_config = {i: as_template.copy() for i in as_ids}
    lb_config = {i: lb_template.copy() for i in lb_ids}

    #Initialization with baseline (desactivated)
    #for i in range(n_worker2change):  # update half as configuration
        #as_config[i].update({'n_worker': n_worker_baseline*n_worker_multiplier})
        
    #Initialization with distribution (desactivated)
    # for i in as_config:
    #     as_config[i].update({'n_worker': int(np.random.choice([1,2,4,8,16], p=n_worker_multiplier_distribution))})
        
    #Initialization with statically set weights
    try:
        if n_as == 64:
            for i in range (8):
                for j in range (8):    
                    as_config[8*i+j].update({'n_worker': i+1})     
        if n_as == 16:
            for i in range (4):
                for j in range (4):    
                    as_config[4*i+j].update({'n_worker': i+1})     
        if n_as == 4:
            for i in range (2):
                for j in range (2):    
                    as_config[2*i+j].update({'n_worker': i+1})     
    except:
        pass 
    
    
    if 'config' in METHODS[lb_method].keys():
        if 'weights' in METHODS[lb_method]['config'].keys() and METHODS[lb_method]['config']['weights'] == {}:
            METHODS[lb_method]['config']['weights'] = {
                    i: as_config[i]['n_worker'] for i in as_ids}
        for i in lb_config.keys():
            lb_config[i].update(METHODS[lb_method]['config'])
    if 'rlb' in lb_method:
        for i in lb_config.keys():
            lb_config[i].update({'logger_dir': log_folder+'/rl.log',
                                 'rl_test': rl_test})

    return {
        'clt': clt_config,
        'er': er_config,
        'as': as_config,
        'lb-'+lb_method: lb_config,
    }

NODE_CONFIG = {}

# ---------------------------------------------------------------------------- #
#                             Control Plane Events                             #
# ---------------------------------------------------------------------------- #

CP_EVENTS2ADD = [
    # (ts, event_name, added_by, **kwargs)
    # e.g.:
    # (
    #     # change second 1/4 AS nodes back to normal worker baseline
    #     200.0+1e-7,
    #     'as_update_capacity',
    #     'sys-admin',
    #     {
    #         'node_ids': ['as{}'.format(i) for i in range(32, 64)],
    #         'n_worker': N_WORKER_BASELINE,
    #         'mp_level': 1,
    #     }
    # ),
    # (
    #     200.0+2e-7,
    #     'as_update_capacity',
    #     'sys-admin',
    #     {
    #         'node_ids': ['as{}'.format(i) for i in range(64, 96)],
    #         'n_worker': N_WORKER_BASELINE*N_WORKER_MULTIPLIER,
    #         'mp_level': 1,
    #     }
    # ),
    # (
    #     # change second 1/4 AS nodes back to normal worker baseline
    #     400.0+1e-7,
    #     'as_update_capacity',
    #     'sys-admin',
    #     {
    #         'node_ids': ['as{}'.format(i) for i in range(32, 64)],
    #         'n_worker': N_WORKER_BASELINE*N_WORKER_MULTIPLIER,
    #         'mp_level': 1,
    #     }
    # ),
    # (
    #     800.0+2e-7,
    #     'as_update_capacity',
    #     'sys-admin',
    #     {
    #         'node_ids': ['as{}'.format(i) for i in range(6,8)],
    #         'n_worker': 1,
    #         'mp_level': 1,
    #     }
    # ),
    # (
    #     1200.0+2e-7,
    #     'lb_remove_server',
    #     'sys-admin',
    #     {
    #         'lbs': [0],
    #         'ass': [6,7]
    #     }
    # ),
    #     (
    #     400.0+2e-7,
    #     'lb_add_server',
    #     'sys-admin',
    #     {
    #         'lbs': [0],
    #         'ass': [32,33,34,35,36,37,38,39],
    #     }
    # ),
    # (
    #     6.,
    #     'lb_remove_server',
    #     {
    #         'lbs': [0],
    #         'ass': [17, 18]
    #     }
    # ),
    # (
    #     3.2,
    #     'clt_update_in_traffic',
    #     {
    #         'node_id': 'clt1',
    #         'in_traffic_info_new': {'rate': 20, 'type': 'normal', 'mu': 1.0, 'std': 0.3}
    #     }
    # )
    (
        
        0.5,
        'as_periodic_log',
        'sys-admin',
        {
            #'node_ids': ['as{}'.format(i) for i in range(64)],
            'node_ids': None,
            'interval': 0.5,
        }
    ),
]
