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

import shm_proxy as sm
from env import *
import argparse
import struct
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker


#--- Initialization ---#

def softmax(x):
    '''
    Compute softmax values given sets of scores in x.
    '''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Heuristic():

        
    def __init__(self, feature_name, map_func, action_range=1., alpha=0.5, logger=None):
        '''
        @brief:
            This class generate actions (weights) based on features gathered by reservoir sampling
        @param:
            feature_name: the feature name that we use to calculate weights (choose amongst sm.FEATURE_AS_ALL)
            map_func: map function e.g. reciprocal or negative
            action_range: by default 1.
            alpha: parameter for soft weights update
            logger: logging info
        '''
        self.n_as = sm.GLOBAL_CONF['global']['SHM_N_BIN']  # n_as = num_actions
        self.feature_idx = sm.FEATURE_AS_ALL.index(feature_name)
        self.map_func = map_func
        self.alpha = alpha
        self.logger = logger
        self.action_range=action_range

    def calculate_weight(self, feature_as, active_as):
        '''
        @param:
            feature_as: a numpy matrix w/ shape (n_as, n_feature_as)
            active_as: a list of current active application server id
        '''
        weights = np.zeros(self.n_as)

        feature = feature_as[active_as, self.feature_idx]

        weights[active_as] = softmax(self.map_func(feature))

        return weights

    def get_action(self, state, last_action):
        '''
        @return:
            action: w/ shape [num_actions]
        '''

        active_as, feature_as, _ = state  # ignore gt

        if len(active_as) == 0: return np.ones(self.n_as)

        weights = self.alpha * self.calculate_weight(feature_as, active_as) + (1-self.alpha) * last_action
        
        # a mask that leaves only active AS's action
        action_mask=np.zeros_like(weights)
        action_mask[active_as]=self.action_range
        action=action_mask * weights

        return action

    def get_init_action(self, active_as):
        '''
        @return:
            action: w/ shape [#num_actions]
        '''
        action = np.ones(self.n_as)
        # a mask that leaves only active AS's action
        action_mask = np.zeros_like(action)
        if len(active_as) == 0:
            return action
        action_mask[active_as] = self.action_range
        return action_mask * action

#--- Arguments ---#

parser=argparse.ArgumentParser(
    description = 'Load Balance Environment w/ Openai Gym APIs.')

parser.add_argument('-v', action = 'store_true',
                    default = False,
                    dest = 'verbose',
                    help = 'Set verbose mode and print out all info')

parser.add_argument('-d', action = 'store_true',
                    default = False,
                    dest = 'dev',
                    help = 'Set dev mode and test offline without opening shared memory file')

parser.add_argument('-m', action = 'store_true',
                    default = 'alias',
                    dest = 'method',
                    help = 'Set method to encode action [\'alias\' for weighted-sampling, \'score\' for deterministic evaluation]')

parser.add_argument('-i', action = 'store',
                    default = 0.5,
                    dest = 'interval',
                    help = 'Set sleep interval in env.step() for action to take effect')

parser.add_argument('-g', action='store_true',
                    default=False,
                    dest='gt',
                    help='Set if collect ground truth')

parser.add_argument('--version', action = 'version',
                    version = '%(prog)s 1.0')

#--- Macros ---#
frame_idx=0  # number of iterations
max_steps=9000  # for dev
render_cycle=2  # every ${render_cycle} steps, print out once current state
action_range=1.
action_dim=sm.GLOBAL_CONF['global']['SHM_N_BIN']  # n_as
feature_name = "flow_duration_avg_decay"
map_func = lambda x: -x # option 1: negative
# map_func = lambda x: 1/x # option 2: reciprocal
rewards=[]

if __name__ == '__main__':
    logger=init_logger("rl/rl.log", "rl-logger")

    args=parser.parse_args()

    lbenv = LoadBalanceEnv(args.interval, logger,
                           verbose=args.verbose, gt=args.gt)
    state=lbenv.reset()
    heuristic = Heuristic(feature_name, map_func, logger=logger)
    last_action = heuristic.get_init_action(state[0])


    for step in range(max_steps):
        action=heuristic.get_action(state, last_action)

        if (action > 0).any(): # at least one AS has non-zero weight 
            next_state, reward, _, info=lbenv.step(action)
        else:
            logger.info(">> no action > 0 ({}), sleep for {}".format(action, args.interval))
            time.sleep(args.interval)
            continue

        state = next_state
        last_action = action
        frame_idx += 1

        # render
        if frame_idx % render_cycle == 0:
            lbenv.render()
