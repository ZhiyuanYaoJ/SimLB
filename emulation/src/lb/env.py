#--- Import ---#
import logging
import sys
import datetime
from os import path

import math
import time
import random
import numpy as np
import gym
from gym import spaces

from scipy.linalg import block_diag

import shm_proxy as sm
import argparse
import struct

#--- MACROS ---#

ACTION_TYPE_OPTIONS = ['score', 'alias']
# Regret version, e.g. -(weighted_average(mean(FCT))), clip value to avoid dramatic gradient
REGRET_RANGE = (-5, 0)
# Reward version, e.g. fairness, clip value to avoid dramatic gradient
REWARD_RANGE = (0, 1)
SEQ_LEN = 8  # How many time steps do we look back into the history
# Maximum value in the feature range, so as to avoid seeing NaNs w/ np.inf
MAX_FEATURE_VALUE = np.power(2., 16)
TEMPORAL_RANGE = (0., MAX_FEATURE_VALUE)  # Temporal feature range
COUNTER_RANGE = (0., MAX_FEATURE_VALUE)  # Counter feature range
WIN_RANGE = (0, 65535)  # Window size feature range
DWIN_RANGE = (-32768, 32768)  # 1st derivative of window size feature range
BYTE_PACKET_RANGE = (0, 1500)  # Byte transmitted per packet feature range
# Byte transmitted per flow feature range
BYTE_FLOW_RANGE = (0, MAX_FEATURE_VALUE)
MAX_FEATURE_VALUE = np.power(2., 16)
TEMPORAL_RANGE = (0., MAX_FEATURE_VALUE)  # Temporal feature range
COUNTER_RANGE = (0., MAX_FEATURE_VALUE)  # Counter feature range
WIN_RANGE = (0, 65535)  # Window size feature range
DWIN_RANGE = (-32768, 32768)  # 1st derivative of window size feature range
BYTE_PACKET_RANGE = (0, 1500)  # Byte transmitted per packet feature range
# Byte transmitted per flow feature range
BYTE_FLOW_RANGE = (0, MAX_FEATURE_VALUE)
DISCRETE_ACTION_SPACE = [1., 1.5, 2.0] # action space

def init_logger(filename, logger_name):
    '''
    @brief:
        initialize logger that redirect info to a file just in case we lost connection to the notebook
    @params:
        filename: to which file should we log all the info
        logger_name: an alias to the logger
    '''

    # get current timestamp
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(filename=filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Test
    logger = logging.getLogger(logger_name)
    logger.info('### Init. Logger {} ###'.format(logger_name))
    return logger

def calcul_fair(values):
    '''
    @brief:
        calculate fairness
    @params:
        values: a list of values
    '''
    values = np.array(values)
    n = len(values)
    if sum(values) != 0.:
        return pow(sum(values), 2)/(n*sum(pow(values, 2)))
    else:
        return 1.

def calcul_fair_bossaer(values, k=1):
    '''
    @brief:
        calculate G's fairness
    @params:
        values: a list of values
        k: kth order fairness
    '''
    values = np.array(values)
    return np.prod(np.power(values/(max(values)+1e-6), 1/k))

#--- Wrapper ---#

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.) * .5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action

#--- Environment ---#

class LoadBalanceEnv(gym.Env):
    '''
    @brief:
        An environment for load-aware load-balancing respecting openai gym API.
    @state:
        States are a (multi-segment averaged) series of LB local observations that are fetched from shared memory with a fixed frequency.
    @action:
        Action space is a list of softmax weights over all servers. [TBD]
    @reward:
        Reward can be calculated according to flow complete time and/or flow duration gathered in reservoir sampling.
    '''
    metadata = {'render.modes': ['cli']}

    def __init__(self, interval, logger=None, verbose=False, gt=False, reward_field='flow_duration_avg_decay', momentum=0.1, discrete=False):
        '''
        @brief:
            Initialize interfaces 
        @params:
            interval: wait a bit to gather reward when taking one step
            logger: a logger that dumps info to a text file
        '''
        super(LoadBalanceEnv, self).__init__()
        # initialize sleep interval
        assert interval > 0
        self.interval = interval
        self.logger = logger
        self.gt = gt
        self.t0 = time.time()
        # Get communication API w/ shared memory
        self.shm_m = sm.Shm_Manager(sm.CONF_FILE, verbose=verbose, gt=gt)

        self.reward_field = reward_field
        self.reward_field_idx = sm.FEATURE_AS_ALL.index(self.reward_field)

        # Actions of the format [score (or weight)] * N_AS
        self.discrete = discrete
        if discrete:
            self.action_space = DISCRETE_ACTION_SPACE
            self.num_actions = len(self.action_space)

        # use exponential moving average to make reward more stable
        self.reward_field_values = np.array([-1]*sm.GLOBAL_CONF["global"]["SHM_N_BIN"])
        self.reward_momentum = momentum

        self.n_feature = len(sm.FEATURE_AS_ALL)
        # initialize
        if self.gt:
            self.gt_sockets = sm.get_sockets(sm.HOST, sm.PORT)
        self.info_buffer = {}

    def reset(self):
        '''
        @brief:
            Reset the state of the environment to an initial state
        '''
        # self.shm_m.clear_shm()

        # initialize log
        self.log = {
            'timestamp': [],
            'obs': [],
            'action': [],
            'reward': [],
            # 'overprovision': [0.], # register a 0 to calculate reward for the first step
        }
        self.current_step = 0
        obs = self._next_observation()

        return obs

    def _next_observation(self):
        '''
        @brief:
            get the data points stored in feature buffers for all servers w/ shape [N_ACTIVE_AS, SEQ_LEN, N_FEATURES]
        @return:
            a tuple that contains (list of active AS id, feature gathered on LB node, feature gatherd on AS nodes as a 2d array, ground truth observation obtained directly from AS node), where the last element (ground truth) may be None if self.gt is set as False
        '''
        return self.shm_m.get_latest_frame()

    def _take_action(self, action, active_as=None):
        '''
        @brief:
            set the weights in LB node according to action_type ('score' - the lower the better, or 'alias' weighted sampling)
        @note:
            in action, not only the weights for each AS are encoded, but also the active AS IDs are indexed as well.
        '''
        final_action = np.zeros(self.shm_m.shm_n_bin)
        if self.discrete:
            action = [self.action_space[a] for a in action]
        if active_as:
            final_action[active_as] = action
        else:
            final_action = action
        self.shm_m.register_as_weights(self.current_step, final_action)

    def _calcul_reward(self, active_as, feature_as, gt=None):
        '''
        @brief:
            calculate reward based on the fairness of recent reservoir observation (flow duration avg)
        '''
        # use negative reservoir_flow_duration_avg_decay times their corresponding AS weights (weighted average) as reward for now
        # previous_active_as = [i for i, v in enumerate(action) if v > 0]
        # asid_idx = [] # a list of persistent asid index in old list 
        # asid_idx_new = [] # a list of persistent asid index in new list 
        # for i, asid in enumerate(previous_active_as):
        #     try: # if the previous AS still within the current as list
        #         i_new = list(active_as).index(asid)
        #         asid_idx.append(i)
        #         asid_idx_new.append(i_new)
        #     except:
        #         continue

        # if len(asid_idx_new) > 0:
        reward_field_values = []
        for asid in active_as:
            # asid_actual = active_as[asid]
            reward_value = self.reward_field_values[asid]
            feature_as_value = feature_as[asid, self.reward_field_idx]
            if reward_value == -1:
                reward_field_value = feature_as_value
            else:
                reward_field_value = (1-self.reward_momentum)*reward_value+self.reward_momentum*feature_as_value
            self.reward_field_values[asid] = reward_field_value
            reward_field_values.append(reward_field_value)
            # self.logger.info("env: asid {} - feature_as_value {}, reward_field_value {}".format(asid, feature_as_value, reward_field_value))
        
        '''
        >> option 1: current overprovision
        '''
        # overprovision = 1-max(reward_field_values)/(np.mean(reward_field_values)+1e-6)  # 1 - overprovision so that the best situation yields 0 instead of -1
        # reward = overprovision

        '''
        >> option 2: difference between current overprovision and last evaluated reward
        '''
        # reward = overprovision - self.log['overprovision'][-1]
        
        # print(">> @_calcul_reward: reward={} reward_field_values={}, action={}, previous_active_as={}, active_as={}, asid_idx={}".format(
            # reward, reward_field_values, action, previous_active_as, active_as, asid_idx))

        # self.log['overprovision'].append(overprovision)

        '''
        >> option 3: fairness
        '''
        reward = calcul_fair(reward_field_values) - 1

        # else:
        #     # print(">> @_calcul_reward: asid_idx_new length=0, previous_active_as={}, active_as={}, asid_idx={}".format(previous_active_as, active_as, asid_idx))
        #     reward = 0

        self.info_buffer = {"reward_fields": reward_field_values}
        # return -(np.array(reward_fields) * np.array(action)[active_as[asid_idx]]), (asid_idx, asid_idx_new)
        return reward

    def sample_action(self, num_actions):
        if self.discrete:
            action = np.random.randint(0, high=self.num_actions, size=num_actions)
        else:
            action = np.random.randn(num_actions)
        return action

    def step_take_action(self, action, active_as):
        
        self._take_action(action, active_as)
        self.current_step += 1
        self.log['action'].append(action)

    def step_get_next_obs(self):
        # get next states and reward
        obs = self._next_observation()
        reward = self._calcul_reward(*obs)

        self.log['timestamp'].append(time.time())
        self.log['obs'].append(obs)
        self.log['reward'].append(reward)

        return obs, reward, False, self.info_buffer  # for now done is always False

    def step(self, action, active_as=None):
        '''
        @brief:
            execute one time step within the environment
        @params:
            action: a list of weights 
        '''
        self._take_action(action, active_as=active_as)

        self.current_step += 1

        # sleep for a while
        time.sleep(self.interval)

        # get next states and reward
        obs = self._next_observation()
        reward = self._calcul_reward(*obs)

        self.log['timestamp'].append(time.time())
        self.log['obs'].append(obs)
        self.log['action'].append(action)
        self.log['reward'].append(reward)

        return obs, reward, False, self.info_buffer # for now done is always False

    def dev(self):
        '''
        @brief:
            for off-line dev purpose
        '''
        print("=== Yo! This is dev mode! ===")

    def cleanup(self):
        if self.gt:
            for s in self.gt_sockets:
                s.close()

    def render(self, mode='cli'):
        '''
        @brief:
            Render the environment to the screen (print out the final step for actions)
        '''
        def max_scale(v):
            try:
                _max = max(v)
            except:
                _max = 0
            return np.array(v)/max(1e-6, _max), _max

        if self.gt:
            for s in self.gt_sockets:
                s.sendall(b'42\n')  # send query to AS and wait for response
            gt_lst_apache = [0] * len(self.gt_sockets)
            gt_lst_cpu = [0] * len(self.gt_sockets)

        active_as, feature_as, gt = self._next_observation()
        if self.logger:
            self.logger.info("="*10)
            self.logger.info("{:<30s}".format("Step: {} (Time: {:.3f}s)".format(self.current_step, time.time()- self.t0)))
            # self.logger.info("{:<30s}".format("Latest Reward: ")+"{}".format(self.log['reward'][-1]))

        if self.gt:
            for i, s in enumerate(self.gt_sockets):
                data = s.recv(24)  # send query to AS and wait for response
                gt_ = list(struct.unpack('dqii', data))
                gt_lst_apache[-i-1] = gt_[-2]  # get number of busy apache threads
                gt_lst_cpu[-i-1] = gt_[0]  # get number of busy apache threads
        # gt_scaled, gt_max = max_scale(gt_lst)
        # self.logger.info("{:<30s}".format("Latest max #apache:"), gt_max)
        if mode == 'ipynb':
            if len(active_as) > 0:
                fig = plt.figure(figsize=(16, 4))
                ax = fig.add_subplot(111)
                frame = np.vstack(
                    gt_lst_apache,
                    self.log['action'][-1]/max(self.log['action'][-1]),
                    feature_as[active_as, sm.FEATURE_AS_ALL.index("flow_duration_avg_decay")],
                    feature_as[active_as, sm.FEATURE_AS_ALL.index(
                        "fct_avg_decay")],
                )
                cax = ax.matshow(frame, cmap='bone')
                fig.colorbar(cax)

                active_as.insert(0, 0)
                ax.set_xticklabels(active_as)
                ax.set_yticklabels(
                    ['', '#apache', 'action', 'avg. flow dur. decay', 'avg. fct decay'])

                # Show label at every tick
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

                plt.show()
                return frame
            else:
                print("No active AS")
        elif mode == 'cli':
            if self.logger:
                # self.logger.info("{:<30s}".format("Latest active ASs:")+' |'.join(
                #     [" {:>7d}".format(asid) for asid in active_as]))
                if self.gt:
                    self.logger.info("{:<30s}".format("#apache:")+' |'.join(
                        [" {:>7d}".format(gt) for gt in gt_lst_apache]))
                    self.logger.info("{:<30s}".format("cpu:")+' |'.join(
                        [" {:> 7.3f}".format(gt) for gt in gt_lst_cpu]))
                    self.logger.info("{:<30s}".format("Last action:")+' |'.join(
                        [" {:> 7.3f}".format(_) for _ in np.array(self.log['action'][-1][1:8])]))
                # self.logger.info("{:<30s}".format("Reward field value:")+' |'.join(
                #     [" {:> 7.3f}".format(_) for _ in np.array(self.reward_field_values)[active_as]]))
                # self.logger.info("{:<30s}".format(
                #     "Last reward: {}".format(self.log['reward'][-1])))
                # self.logger.info("{:<30s}".format("Latest avg. flow dur. decay:")+' |'.join([" {:> 7.3f}".format(
                #     _) for _ in feature_as[active_as, sm.FEATURE_AS_ALL.index("flow_duration_avg_decay")]]))
                self.logger.info("{:<30s}".format("Latest avg. fct. decay:")+' |'.join([" {:> 7.3f}".format(
                    _) for _ in feature_as[active_as, sm.FEATURE_AS_ALL.index("fct_avg_decay")]]))
            else:
                print("{:<30s}".format("Latest active ASs:")+' |'.join(
                    [" {:>5d}".format(asid) for asid in active_as]))
                if self.gt:
                    print("{:<30s}".format("#apache:")+' |'.join(
                        [" {:>5d}".format(gt) for gt in gt_lst]))
                if len(self.log['action']) > 0:
                    print("{:<30s}".format("Last action:")+' |'.join(
                        [" {:>0.2e}".format(_) for _ in np.array(self.log['action'][-1])[active_as]]))
                print("{:<30s}".format("Latest avg. flow dur. decay:")+' |'.join([" {:>0.3f}".format(
                    _) for _ in feature_as[active_as, sm.FEATURE_AS_ALL.index("flow_duration_avg_decay")]]))
                print("{:<30s}".format("Latest avg. fct. decay:")+' |'.join([" {:>0.3f}".format(
                    _) for _ in feature_as[active_as, sm.FEATURE_AS_ALL.index("fct_avg_decay")]]))
