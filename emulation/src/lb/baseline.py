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


class Weighted():
        
    def __init__(self, high, low, n_active=8, weights=None, logger=None):
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
        self.high = high
        self.low = low
        self.n_active = n_active
        self.weights = weights
        self.logger = logger


    def get_init_action(self, active_as):
        '''
        @return:
            action: w/ shape [#num_actions]
        '''
        action = np.zeros(self.n_as)

        if len(active_as) == 0:
            active_as = list(range(self.n_active+1, self.n_active*2+1))
        if self.weights is None:
            active_as = range(1+self.n_active, 1+self.n_active*2)
            l_unit = int(len(active_as)/4)
            action[active_as[:l_unit]] = self.high
            action[active_as[l_unit:2*l_unit]] = self.low
            action[active_as[2*l_unit:3*l_unit]] = self.high
            action[active_as[3*l_unit:4*l_unit]] = self.low 
        else:
            action[active_as] = self.weights[::-1]

        return action

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

parser.add_argument('-i', action = 'store_true',
                    default = 0.25,
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
high = 1.
low = 0.5
rewards=[]
action = np.zeros(action_dim) # just take 0 as actions

if __name__ == '__main__':
    logger=init_logger("log/lb.log", "rl-logger")

    args=parser.parse_args()

    lbenv = LoadBalanceEnv(args.interval, logger,
                           verbose=args.verbose, gt=args.gt)
    state = lbenv.reset()

    for step in range(max_steps):

        next_state, reward, _, info=lbenv.step(action)
        
        state = next_state
        frame_idx += 1

        # render
        if frame_idx % render_cycle == 0:
            lbenv.render()
