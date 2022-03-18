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
        
    def __init__(self, active_as_idx, weights, action_dim, active=False, logger=None):
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
        self.action_dim = action_dim
        self.active_as_idx = active_as_idx
        self.weights = weights
        self.logger = logger
        self.init_action = np.zeros(self.action_dim)
        self.init_action[self.active_as_idx] = self.weights[::-1]
        if active:
            self.get_action = self.get_action_active
        else:
            self.get_action = self.action_repeat

    def action_repeat(self, state):
        active_as = state[0]
        assert len(set(active_as) - set(self.active_as_idx)) == 0
        return self.init_action

    def get_action_active(self, state):

        action = np.zeros(self.action_dim)
        gt = state[-1]
        active_as = state[0]
        as_idx = [gt[asid][-1] for asid in active_as]
        # self.logger.info("@get_action_active:\nactive_as:{}\nas_idx:{}".format(active_as, as_idx))
        if len(active_as) > 0:
            load = np.array([gt[asid][2] for asid in active_as])
            weights = self.init_action[active_as]

            # spotlight implementation (sed-like)
            available_cap = weights/(load+1)
            new_weights = available_cap/max(available_cap)
            
        else:
            active_as = self.active_as_idx
            new_weights = self.init_action[self.active_as_idx]
        action[active_as] = new_weights
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


parser.add_argument('-i', action = 'store',
                    default = 0.25,
                    dest = 'interval',
                    type = float,
                    help = 'Set sleep interval in env.step() for action to take effect')

parser.add_argument('-g', action='store_true',
                    default=False,
                    dest='gt',
                    help='Set if collect ground truth')

parser.add_argument('-a', action='store_true',
                    default=False,
                    dest='active',
                    help='Set if active probing is enabled')

parser.add_argument('--version', action = 'version',
                    version = '%(prog)s 1.0')

#--- Macros ---#
frame_idx=0  # number of iterations
max_steps=9000  # for dev
render_cycle=2  # every ${render_cycle} steps, print out once current state
action_range=1.
action_dim=sm.GLOBAL_CONF['global']['SHM_N_BIN']  # n_as
n_active=sm.GLOBAL_CONF['meta']['n_as']
active_as_idx = list(range(1, 1+n_active))
active_weights = sm.GLOBAL_CONF['meta']['weights']
rewards=[]

if __name__ == '__main__':
    logger=init_logger("log/lb.log", "rl-logger")

    args=parser.parse_args()

    lbenv = LoadBalanceEnv(args.interval, logger,
                           verbose=args.verbose, gt=args.gt)
    state = lbenv.reset()
    actor = Weighted(logger=logger, active_as_idx=active_as_idx, weights=active_weights, action_dim=action_dim, active=args.active)

    for step in range(max_steps):
        action=actor.get_action(state)

        next_state, reward, _, info=lbenv.step(action)

        state = next_state
        last_action = action
        frame_idx += 1

        # render
        if frame_idx % render_cycle == 0:
            lbenv.render()
