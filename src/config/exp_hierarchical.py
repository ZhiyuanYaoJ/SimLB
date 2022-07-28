# ---------------------------------------------------------------------------- #
#                                  Description                                 #
# This file stores a hierarchical configuration, which will overwrite default #
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
N_LBP = 1
N_LBS = 6
LBP_BUCKET_SIZE = LB_BUCKET_SIZE
LBS_BUCKET_SIZE = LB_BUCKET_SIZE
N_LAYER = 2

def generate_node_config_hierarchical(
    lbp_method='ecmp',
    lbs_method='ecmp',
    n_clt=1, 
    n_er=1, 
    n_lbp=N_LBP,
    n_lbs=N_LBS,  
    n_as=N_AS, 
    n_worker_baseline=N_WORKER_BASELINE, 
    n_worker2change=N_WORKER2CHANGE, 
    n_worker_multiplier=N_WORKER_MULTIPLIER,
    n_worker_multiplier_distribution=N_WORKER_MULTIPLIER_DISTRIBUTION,
    as_mp_level=AS_MULTIPROCESS_LEVEL,
    lbp_bucket_size=LBP_BUCKET_SIZE,
    lbs_bucket_size=LBS_BUCKET_SIZE,
    log_folder=LOG_FOLDER,
    rl_test=False,
    debug=DEBUG):

    clt_ids = list(range(n_clt))
    er_ids = list(range(n_er))
    lbp_ids = list(range(n_lbp))
    lbs_ids = list(range(n_lbp, n_lbp + n_lbs))
    as_ids = list(range(n_as))

    clt_template = {
        'child_ids': er_ids,
        'child_prefix': 'er',  # connected to edge router
        'debug': 0
    }

    er_template = {
        'child_ids': lbp_ids,
        'child_prefix': 'lb',  # connected to load balancers
    }

    lbp_template = {
        'child_ids': lbs_ids,
        'debug': 0,
        'layer': 2,
        'bucket_size': lbp_bucket_size,
    }
    
    lbs_template = {
        'child_ids': as_ids,
        'debug': 0,
        'layer': 1,
        'bucket_size': lbs_bucket_size,
    }

    as_template = {
        'n_worker': n_worker_baseline,
        'multiprocess_level': as_mp_level,
        'debug': 0
    }

    clt_config = {i: clt_template.copy() for i in clt_ids}
    er_config = {i: er_template.copy() for i in er_ids}
    as_config = {i: as_template.copy() for i in as_ids}
    lbp_config = {i: lbp_template.copy() for i in lbp_ids}
    lbs_config = {i: lbs_template.copy() for i in lbs_ids}
    

    k, m = divmod(n_as, n_lbs)
    if m>0: k+=1
    for i in lbs_config:
        lbs_config[i]['child_ids'] = list(range((i-1)*k,min((i)*k, n_as)))
        
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
        
    # For secondary LB
    if 'config' in METHODS[lbs_method].keys():
        if 'weights' in METHODS[lbs_method]['config'].keys() and METHODS[lbs_method]['config']['weights'] == {}:
            METHODS[lbs_method]['config']['weights'] = {
                    i: as_config[i]['n_worker'] for i in as_ids}
        for i in lbs_config.keys():
            lbs_config[i].update(METHODS[lbs_method]['config'])
    if 'rlb' in lbs_method:
        for i in lbs_config.keys():
            lbs_config[i].update({'logger_dir': log_folder+'/rls.log',
                                'rl_test': rl_test})            


    # For primary LB
    if 'config' in METHODS[lbp_method].keys():
        #If primary LBs need weights (e.g. WCMP) but not secondary LBs, set weights as the sum of child capacities
        if 'weights' in METHODS[lbp_method]['config'].keys() and METHODS[lbp_method]['config']['weights'] == {}:
            METHODS[lbp_method]['config']['weights'] = {
                    i: sum([as_config[k]['n_worker'] for k in lbs_config[i]['child_ids']]) for i in lbs_ids}
            
        #If primary LBs and secondary LBs need weights (e.g. WCMP), set weights2 as the sum of child capacities        
        elif 'weights' in METHODS[lbp_method]['config'].keys() and not METHODS[lbp_method]['config']['weights'] == {}:
            for i in lbp_config:
                lbp_config[i].update({'weights2': {i: sum([as_config[k]['n_worker'] for k in lbs_config[i]['child_ids']]) for i in lbs_ids }})
            
        for i in lbp_config.keys():
            lbp_config[i].update(METHODS[lbp_method]['config'])
            
    if 'rlb' in lbp_method:
        for i in lbp_config.keys():
            lbp_config[i].update({'logger_dir': log_folder+'/rlp.log',
                                 'rl_test': rl_test})

    if lbp_method == lbs_method:
        lb_config = {**lbp_config, **lbs_config}
        return {
            'clt': clt_config,
            'er': er_config,
            'as': as_config,
            'lb-'+lbp_method: lb_config
        }
    else:
        return {
            'clt': clt_config,
            'er': er_config,
            'as': as_config,
            'lb-'+lbp_method: lbp_config,
            'lb-'+lbs_method: lbs_config,
        }
    
NODE_CONFIG = {}

# ---------------------------------------------------------------------------- #
#                             Control Plane Events                             #
# --------------------------------- ------------------------------------------- #

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
    #     400.0+2e-7,
    #     'as_update_capacity',
    #     'sys-admin',
    #     {
    #         'node_ids': ['as{}'.format(i) for i in range(64, 96)],
    #         'n_worker': N_WORKER_BASELINE,
    #         'mp_level': 1,
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
        # Periodic log
        0.5,
        'as_periodic_log_hierarchical',
        'sys-admin',
        {
            #'node_ids': ['as{}'.format(i) for i in range(N_AS)],
            'node_ids': None,
            'interval': 0.5,
        }
    ),
]
