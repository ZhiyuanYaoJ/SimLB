# ---------------------------------------------------------------------------- #
#                                  Description                                 #
# This file contains default global configuration that will be loaded in       #
# simulator                                                                    #
# ---------------------------------------------------------------------------- #

import time
import random
import numpy as np
import argparse

SEED = 42  # random seed
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------- #
#                                     Nodes                                    #
# ---------------------------------------------------------------------------- #

# ------------------------------------ AS ------------------------------------ #

N_AS = 12  # number of application server node(s)

ACTION_DIM = 600  # maximal amount of application server nodes, as in emulator

# baseline number of worker threads for each AS 
N_WORKER_BASELINE = 2

# mutliprocessing level
AS_MULTIPROCESS_LEVEL = 1

# in terms of server capacity variance
N_WORKER2CHANGE = 0.5
N_WORKER_MULTIPLIER = 2
N_WORKER_MULTIPLIER_DISTRIBUTION = [0.2, 0.4, 0.3, 0.1, 0]

AS_MAX_CLIENT = 64

T_TIMEOUT = 40 # rejected flows will be returned to client after 40 seconds

# ------------------------------------ LB ------------------------------------ #

N_LB = 1  # number of load balancer node(s)

# (unit s) load balancing weights update period
LB_PERIOD = 0.5

# Load distribution method (including, random, random-weight, static-weight, lsq, lsq-po2, heuristic, kf1d, rl-sac)
METHOD = 'heuristic'

# the feature to be used to calculate reward | previously as `reward_feature`
REWARD_FEATURE = 'res_fct_avg_disc'
HIDDEN_DIM = 512

# including 
#   0: 1-overprovision; 
#   1: negative exponential difference between mean and max; 
#   2: difference between min and max; 
#   3: exponential difference between min and max; 
#   4: jain's fairness index
REWARD_OPTION = 4

# update padding option 
#   'valid': keep incrementing specific time interval on last weights generation timestamp
#   'same': keep intervals between weights generating timestamps the same
LB_UPDATE_PADDING = 'valid'

LB_BUCKET_SIZE = 65536

# ------------------------------------ Clustering----------------------------- #

HIERARCHICAL = False
CLUSTERING_PERIOD = 0.1
CLUSTERING_METHOD = 'kmeans'

# ---------------------------------- Client ---------------------------------- #

LOAD_OBSERVE_INTERVAL = 0.5 # every 0.5s count #flow on each AS node

# ---------------------------------------------------------------------------- #
#                                   Features                                   #
# ---------------------------------------------------------------------------- #

# (exponential) freshness base number
FRESHNESS_BASE = 0.5

# number of buckets in reservoir sampling buffer
RESERVOIR_BUFFER_SIZE = 32

# probability of replacing an old sample by a new one
RESERVOIR_KEEP_PROB = 1.

# whenever updating reservoir flow duration, how many packets might be transmitted
RESERVOIR_FD_PACKET_DENSITY = 2

# -------------------------------- AS Features ------------------------------- #

# keys of reservoir smaplings process
REDUCE_METHODS = [
    'avg',          # simply calculate average
    'std',          # simply calculate standard deviation
    'p90',          # 90th-percentile
    'avg_disc',     # discounted weighted averaged based on samples' freshness
    'avg_decay',    # simple average based on samples' freshness
    ]

# features that are collected by reservoir sampling in emulator
RESERVOIR_AS_KEYS = [
    'fd',
    'fct',
]

# initialize all collected features for each application server (AS)
FEATURE_AS_ALL = ['n_flow_on']
for f in ['res_{}'.format(k) for k in RESERVOIR_AS_KEYS]:
    FEATURE_AS_ALL += ['{}_{}'.format(f, m) for m in REDUCE_METHODS]

# total amount of features for each AS
N_FEATURE_AS = len(FEATURE_AS_ALL)

# -------------------------------- LB Features ------------------------------- #

# features that are collected for each LB node by reservoir sampling in emulator
RESERVOIR_LB_KEYS = [
    'iat_f_lb',
]

# all features for LB node, including simple averaged features across all its associated AS
FEATURE_LB_ALL = []

for f in ['res_{}'.format(k) for k in RESERVOIR_LB_KEYS]:
    FEATURE_LB_ALL += ['{}_{}'.format(f, m) for m in REDUCE_METHODS]

# total amount of features for each LB node
N_FEATURE_LB = len(FEATURE_LB_ALL) + len(FEATURE_AS_ALL)


# ---------------------------------------------------------------------------- #
#                                    Episode                                   #
# ---------------------------------------------------------------------------- #

N_EPISODE = 1 # number of episodes to run
EPISODE_LEN = 200.  # (unit s) episode length | previously as `args.t_stop`
N_FLOW_TOTAL = None # define this if we don't want to simulate on number of flows instead of episode time
EPISODE_LEN_INC = 1.  # incremental episode length | previously as `args.t_inc`

# ---------------------------------------------------------------------------- #
#                                  Environment                                 #
# ---------------------------------------------------------------------------- #

DEBUG = 0  # level of debug mode 0 < 1 < 2

RENDER = False  # set to False if nothing need to be rendered into a log file every `step`
RENDER_RECEIVE = False  # set to False if nothing need to be rendered into a log file whenever receiving a `flow`

# write to this file, add 'reduce' if we don't need all flows info | previously as `args.log_file`
LOG_FOLDER = 'log'



# ---------------------------------------------------------------------------- #
#                                    Traffic                                   #
# ---------------------------------------------------------------------------- #

PROCESS_N_STAGE = 1

# including normal and exponential distribution of flow complete time (FCT)
CPU_FCT_TYPE = 'exp'

CPU_FCT_MU = 0.5  # average FCT

CPU_FCT_STD = 0.1  # FCT standard deviation (useless for exponential distribution)

CPU_FCT_MIN = 1e-6  # minimal flow complete time

CPU_FCT_MAX = 1. # maximum flow complete time for uniform distribution

IO_FCT_TYPE = 'exp'

IO_FCT_MU = 0.2  # average FCT

IO_FCT_STD = 0.1  # FCT standard deviation (useless for exponential distribution)

IO_FCT_MIN = 1e-6  # minimal flow complete time

IO_FCT_MAX = 1.  # maximum flow complete time for uniform distribution

# for normal distribution of FCT, (assert FCT_MU - FCT_STD_CLIP_LEVEL*FCT_STD > 0) since normal distribution will clip FCT at FCT_MU-FCT_STD_CLIP_LEVEL*FCT_STD
FCT_STD_CLIP_LEVEL = 2

FCT_MIN = 1e-6 # general minimum fct

# normalized poisson traffic rate | previously as `poisson_lambda`
TRAFFIC_RATE_NORM = 0.9
TRAFFIC_RATE = TRAFFIC_RATE_NORM * N_AS * N_WORKER_BASELINE / CPU_FCT_MU

APPLICATION_CONFIG_TEMPLATE = {
    'rate': TRAFFIC_RATE,
    'n_stage': PROCESS_N_STAGE,
    'cpu_distribution': {
        'fct_type': CPU_FCT_TYPE,
        'mu': CPU_FCT_MU
    },
    'io_distribution': {
        'fct_type': IO_FCT_TYPE,
        'mu': IO_FCT_MU
    },
}

def get_app_config(
    rate=None, process_n_stage=None,
    cpu_fct_type=None, cpu_fct_mu=None, cpu_fct_std=None,
    io_fct_type=None, io_fct_mu=None, io_fct_std=None,
    ):
    config = {}
    if rate: config['rate'] = rate
    if process_n_stage: config['n_stage'] = process_n_stage
    if cpu_fct_type:
        config['cpu_distribution'] = {}
        config['cpu_distribution']['fct_type'] = cpu_fct_type
        if cpu_fct_type == 'exp':
            assert cpu_fct_mu
            config['cpu_distribution'].update({'mu': cpu_fct_mu})
        elif cpu_fct_type in ['normal', 'uniform', 'lognormal']:
            assert cpu_fct_mu and cpu_fct_std
            config['cpu_distribution'].update({'mu': cpu_fct_mu, 'std': cpu_fct_std})
        else:
            raise NotImplementedError
    if process_n_stage > 1: # we only care about io part if an application has more than 1 stages
        if io_fct_type:
            config['io_distribution'] = {}
            config['io_distribution']['fct_type'] = io_fct_type
            if io_fct_type == 'exp':
                assert io_fct_mu
                config['io_distribution'].update({'mu': io_fct_mu})
            elif io_fct_type == 'normal':
                assert io_fct_mu and io_fct_std
                config['io_distribution'].update({'mu': io_fct_mu, 'std': io_fct_std})
            elif io_fct_type == 'uniform':
                config['io_distribution'].update({'low': io_fct_low, 'high': io_fct_std})
            else:
                raise NotImplementedError

    return config

# ---------------------------------------------------------------------------- #
#                                   Policies                                   #
# ---------------------------------------------------------------------------- #

# --------------------------------- Heuristic -------------------------------- #

HEURISTIC_ALPHA = 0.5
HEURISTIC_FEATURE = 'res_fd_avg'

# ----------------------------- 1D Kalman Filter ----------------------------- #

KF_CONF = {
    'system_std': 0.01, # system error
    'sensor_std': 0.4, # sensor error
    'system_mean': 0,  # system mean
    'init_mean': 0.5,  # initial state mean
    'init_std': 10.,  # initial state mean
}

# --------------------------------- B Offset --------------------------------- #

B_OFFSET = 1

# ------------------------------ Active Probing ------------------------------ #

# minimal and maximal RTT between AS and LB
RTT_MIN = 1e-3
RTT_MAX = 1e-2

# ---------------------------------------------------------------------------- #

def update_config_w_args(args):
    '''
    @brief:
        update global configuration with system arguments
    '''

    if FCT_TYPE == 'normal':
        assert FCT_MU - FCT_STD_CLIP_LEVEL*FCT_STD > 0
    raise NotImplementedError

# ---------------------------------------------------------------------------- #
#                               Argument Parser                                #
# ---------------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description='Load Balancer Simulator'
)

# ---------------------------------- Episode --------------------------------- #

parser.add_argument('-t', type=float, action='store',
                    default=EPISODE_LEN, dest='t_stop',
                    help='Episode length')

parser.add_argument('--n-flow', type=int, action='store',
                    default=N_FLOW_TOTAL, dest='n_flow_total',
                    help='Total amount of flows')

parser.add_argument('--n-episode', type=int, action='store',
                    dest='n_episode', default=N_EPISODE, help='Number of episodes')

parser.add_argument('--first-episode-id', type=int, action='store',
                    dest='first_episode_id', default=0, help='First episode id starting from?')

parser.add_argument('--t-inc', type=float, action='store',
                    default=EPISODE_LEN_INC, dest='t_inc', help='Incremental episode length')

# -------------------------------- Environment ------------------------------- #

parser.add_argument('-w', action='store', default='test-reduce',
                    dest='log_folder',
                    help='Write to this log file, add \'reduce\' if all flows informations are not required')

parser.add_argument('-m', action='store', default='ecmp', dest='method',
                    help='Load distribution method (ecmp, weight, lsq, lsq2, heuristic, kf1d, sac, ...)')

parser.add_argument('-m1', action='store', default='ecmp', dest='method1',
                    help='Load distribution method (ecmp, weight, lsq, lsq2, heuristic, kf1d, sac, ...)')

parser.add_argument('-m2', action='store', default='ecmp', dest='method2',
                    help='Load distribution method (ecmp, weight, lsq, lsq2, heuristic, kf1d, sac, ...)')

parser.add_argument('--auto-clustering', action='store', default=False, dest='auto_clustering',
                    help='Wether ass should be regrouped among equal size')

parser.add_argument('--dump-all', action='store_true', default=False, dest='dump_all_flow',
                    help='Whether dump all the flows in a file')


# ---------------------------------- Traffic --------------------------------- #

parser.add_argument('--lambda', type=float, action='store', default=TRAFFIC_RATE_NORM,
                    dest='poisson_lambda',
                    help='Normalized poisson traffic rate')

parser.add_argument('--process-n-stage', type=int, action='store', default=PROCESS_N_STAGE,
                    dest='process_n_stage', help='Total amount of stages')

parser.add_argument('--cpu-fct-type', action='store', default=CPU_FCT_TYPE,
                    dest='cpu_fct_type', help='Type of FCT distribution (normal, exp)')

parser.add_argument('--cpu-fct-mu', type=float, action='store',
                    default=CPU_FCT_MU, dest='cpu_fct_mu', help='Average FCT (s)')

parser.add_argument('--cpu-fct-std', type=float, action='store', default=CPU_FCT_STD,
                    dest='cpu_fct_std', help='Normal distribution FCT (s) standard deviation')

parser.add_argument('--io-fct-type', action='store', default=IO_FCT_TYPE,
                    dest='io_fct_type', help='Type of FCT distribution (normal, exp)')

parser.add_argument('--io-fct-mu', type=float, action='store',
                    default=IO_FCT_MU, dest='io_fct_mu', help='Average FCT (s)')

parser.add_argument('--io-fct-std', type=float, action='store', default=IO_FCT_STD,
                    dest='io_fct_std', help='Normal distribution FCT (s) standard deviation')

# --------------------------------- Features --------------------------------- #

parser.add_argument('--fresh-base', type=float, action='store', default=FRESHNESS_BASE,
                    dest='freshness_base', help='Base number that is used to calculate freshness')

# --------------------------------- Policies --------------------------------- #

# Heuristic

parser.add_argument('--heuristic-feature', action='store', default=HEURISTIC_FEATURE,
                    dest='heuristic_feature', help='Choose one feature as observation to estimate server load')

parser.add_argument('--heuristic_alpha', type=float, action='store', default=HEURISTIC_ALPHA,
                    dest='heuristic_alpha', help='Soft update weight')

# 1D Kalman Filter

parser.add_argument('--kf-sys-std', type=float, action='store',
                    default=KF_CONF['system_std'], dest='kf_system_std', help='Kalman Filter system std')

parser.add_argument('--kf-sensor-std', type=float, action='store',
                    default=KF_CONF['sensor_std'], dest='kf_sensor_std', help='Kalman Filter sensor std')

# B method

parser.add_argument('--b-offset', type=int, action='store',
                    default=B_OFFSET, dest='b_offset', help='B offset')


parser.add_argument('--lb-period', type=float, action='store',
                    default=LB_PERIOD, dest='lb_period', help='Periodic weights update interval')

# RL

parser.add_argument('--rl-test', dest='rl_test', action='store_true',
                    default=False, help='Test the trained RL policy')

# ----------------------------------- Nodes ---------------------------------- #

parser.add_argument('--n-clt', type=int, action='store',
                    default=1, dest='n_clt',
                    help='Number of client nodes')

parser.add_argument('--n-er', type=int, action='store',
                    default=1, dest='n_er',
                    help='Number of edge router nodes')

parser.add_argument('--n-lb', type=int, action='store',
                    default=N_LB, dest='n_lb',
                    help='Number of LB nodes')

parser.add_argument('--n-lbp', type=int, action='store',
                    default=1, dest='n_lbp',
                    help='Number of LB nodes in the first layer')

parser.add_argument('--n-lbs', type=int, action='store',
                    default=N_LB, dest='n_lbs',
                    help='Number of LB nodes in the second layer')

parser.add_argument('--n-as', type=int, action='store',
                    default=N_AS, dest='n_as',
                    help='Number of AS nodes')

parser.add_argument('--n-worker', type=int, action='store', default=N_WORKER_BASELINE,
                    dest='n_worker',
                    help='Number of worker threads for each AS')

parser.add_argument('--n-worker-multiplier', type=int, action='store',
                    default=N_WORKER_MULTIPLIER, dest='n_worker_multiplier',
                    help='Change the level of server capacity variance')

parser.add_argument('--as-mp-lv', type=int, action='store',
                    default=AS_MULTIPROCESS_LEVEL, dest='as_mp_level',
                    help='Change the level of server capacity variance')

parser.add_argument('--n-worker2change', type=int, action='store',
                    default=N_WORKER2CHANGE, dest='n_worker2change',
                    help='How many servers\' capacities should be multiplied')

parser.add_argument('--reward-feature', action='store', default=REWARD_FEATURE,
                    dest='reward_feature', help='Use which feature to calculate reward.')

parser.add_argument('--reward-option', type=int, action='store', default=REWARD_OPTION,
                    dest='reward_option', help='Use which way of calcualting reward.')

parser.add_argument('--lb-bucket-size', type=int, action='store',
                    default=LB_BUCKET_SIZE, dest='lb_bucket_size',
                    help='Change the size of flow table')
