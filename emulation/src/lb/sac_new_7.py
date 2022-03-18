#--- Import ---#
import logging
import sys
import datetime
from os import path
from collections import namedtuple

import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import shm_proxy as sm
from env import *
import argparse


#--- MACROS ---#

# Max number of application servers
MAX_N_AS = sm.GLOBAL_CONF["global"]["SHM_N_BIN"]
device = 'cpu'

#--- Initialization ---#


def linear_weights_init(m):
    if isinstance(m, nn.Linear):
        stdv = 1. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)


class ReplayBuffer(object):
    '''
    @brief:
        A simple implementation of ring buffer storing tuple (observation, action, rewards, next_observation)
    @note:
        observation is a tuple that consists of (feature_lb, feature_as, active_as, gt), and each batch will be stored independently in the ring buffer.
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        '''
        @params:
            state/next_state: tuple of (active_as, feature_lb, feature_as, gt)
            action: 7 server weights
            reward: scalar
        @dev:
            for now we don't consider gt at all w/ RL
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (  # ([feature_as reshaped], next_feature_as, action, reward)
            state,
            next_state,
            action,
            reward
        )
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, next_state, action, reward = map(
            np.stack, zip(*batch))

        
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


class SoftQNetwork(nn.Module):
    '''
    @brief:
        evaluate Q value given a state and the action
    '''
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        print("X shape is: {}".format(x.shape))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2, logger=None):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.logger = logger

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range

    def forward(self, state):
        '''
        @param:
            state: torch.FloatTensor w/ shape [#batch, #n_feature]
        '''
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        '''
        @brief:
            generate sampled action with state as input wrt the policy network
        '''

        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        # tanh distribution as actions, reparamerization trick
        action0 = torch.tanh(mean + std*z.to(device))
        action = action0 + 1

        log_prob = Normal(mean, std).log_prob(mean + std*z.to(device)) - \
            torch.log(1. - action0.pow(2) + epsilon) - \
            np.log(self.action_range)

        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic):
        '''
        @return:
            state: w/ shape [#batch, n_feature_as*num*actions]
            action: w/ shape [num_actions]
        '''

        state = torch.FloatTensor(state).to(device)

        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()

        # dev
        # self.logger.info("{:<30s}".format("Mean:")+' |'.join(
        #     [" {}".format(_) for _ in mean.detach().cpu().numpy()]))
        # self.logger.info("{:<30s}".format("Std:")+' |'.join(
        #     [" {}".format(_) for _ in std.detach().cpu().numpy()]))

        # a mask that leaves only active AS's action
        if deterministic:
            action = torch.tanh(mean) + 1
        else:
            action = torch.tanh(mean + std*z) + 1

        return action.detach().cpu().numpy()[0]

    def sample_action(self, active_as):
        '''
        @return:
            action: w/ shape [#num_actions]
        '''
        action = torch.FloatTensor(len(active_as)).uniform_(0, 1)
        return action.numpy()

#--- SAC Trainer ---#


class SAC_Trainer():
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim, action_range, logger=None):
        self.replay_buffer = replay_buffer

        # if DEBUG:
        #     print("state_dim {}, action_dim {}, hidden_dim {}".format(
        #         state_dim, action_dim, hidden_dim))

        self.soft_q_net1 = SoftQNetwork(
            state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(
            state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(
            state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(
            state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(
            state_dim, action_dim, hidden_dim, action_range, logger=logger).to(device)
        self.log_alpha = torch.zeros(
            1, dtype=torch.float32, requires_grad=True, device=device)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr = 3e-4

        self.soft_q_optimizer1 = optim.Adam(
            self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(
            self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        state, action, reward, next_state = self.replay_buffer.sample(
            batch_size)

        
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        # reward is scalar, add 1 dim to be [reward] at the same dim
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(
            state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(
            next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)
                                 ) / (reward.std(dim=0) + 1e-6)

    # update alpha wrt entropy
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob +
                                             target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

    # update Q function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action), self.target_soft_q_net2(
            next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + gamma * target_q_min
        q_value_loss1 = self.soft_q_criterion1(
            predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(
            predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

    # update Policy
        predicted_new_q_value = torch.min(self.soft_q_net1(
            state, new_action), self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path+'_q1')
        torch.save(self.soft_q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path+'_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path+'_q2'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

class RLB_SAC():

        
    def __init__(
        self, 
        state_dim,
        hidden_dim,
        action_dim,
        action_range=1,
        replay_buffer_size=3000,
        auto_entropy=True,
        deterministic=False,
        logger=None
        ):
        '''
        @brief:
            This class generate actions (weights) based on features gathered by reservoir sampling
        @param:
            feature_name: feature keyword that we use to calculate weights (choose among sm.FEATURE_AS_ALL)
            map_func: map function e.g. reciprocal or negative
            action_range: by default 1, then SAC model generates action in range (-1, 1) with tanh function, then we need to add 1 to get weights in (0, 2)
            logger: logging info
        '''
        self.n_as = sm.GLOBAL_CONF['global']['SHM_N_BIN']  # n_as = num_actions
        # set parameters
        self.logger = logger
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.auto_entropy = auto_entropy
        self.deterministic = deterministic
        self.state_buffer = None
        self.action_buffer = None
        self.bn_running_mean = None
        self.bn_running_std = None
        self.bn_momentum = 0.1

        # initialize model
        self.sac_trainer = SAC_Trainer(self.replay_buffer, state_dim, action_dim, hidden_dim, action_range, logger)

    def save(self, model_path):
        self.sac_trainer.save_model(model_path)

    def load(self, model_path):
        self.sac_trainer.load_model(model_path)

    def get_action(self, active_as, feature_as, random_action=False):
        '''
        @return:
            action: w/ shape [#active_as]
        '''

        # register state
        self.state_buffer = feature_as.reshape(-1)

        if len(active_as) == 0: 
            action = []

        if random_action: 
            action = self.sac_trainer.policy_net.sample_action(active_as)
        else:
            action = self.sac_trainer.policy_net.get_action(
                feature_as, deterministic=self.deterministic).reshape(-1)
        # register action
        self.action_buffer = action

        final_action = np.zeros(self.n_as)
        final_action[active_as] = action

        return final_action


    def push_replay_buffer(self, reward, next_state):
        '''
        @brief: push arguments into replay buffer
        '''
        self.replay_buffer.push(self.state_buffer, self.action_buffer, reward, next_state.reshape(-1))

    
    def update_model(self, batch_size, update_iter=1):
        if len(self.replay_buffer) > batch_size:
            for _ in range(update_iter):
                self.sac_trainer.update(
                    batch_size,
                    reward_scale=1,
                    auto_entropy=self.auto_entropy,
                    target_entropy=-1.,
                )


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
                    default = 0.25,
                    dest = 'interval',
                    help = 'Set sleep interval in env.step() for action to take effect')

parser.add_argument('-g', action='store_true',
                    default=False,
                    dest='gt',
                    help='Set if collect ground truth')

parser.add_argument('-t', action='store_false',
                    default=True,
                    dest='train',
                    help='Set to False if no need to train')

parser.add_argument('--version', action = 'version',
                    version = '%(prog)s 1.0')

#--- Macros ---#
frame_idx=0  # number of iterations
batch_size = 64 # number of batches for training
hidden_dim = 128 # number of hidden units
action_dim = 7 # number of actions (active servers)
explore_steps = 100 # number of iterations for pure exploration
update_itr = 1 # how many iterations do we update each time
max_steps=9000  # for dev
render_cycle=2  # every ${render_cycle} steps, print out once current state
save_cycle = 100 # every ${save_cycle} steps, save once current model
action_range=1.
feature_idx = [i for i, f in enumerate(sm.FEATURE_AS_ALL) if 'flow_duration' in f] # feature to be used as input
rewards=[]
model_path = 'rl/sac_new'
# DEBUG = True

if __name__ == '__main__':
    logger=init_logger("log/logger.log", "heuristic-logger")

    args=parser.parse_args()

    lbenv = LoadBalanceEnv(args.interval, logger,
                           verbose=args.verbose, gt=args.gt)
    state=lbenv.reset() # state consists of (active_as, feature_as, gt)
    rlb = RLB_SAC(len(feature_idx)*action_dim, hidden_dim, action_dim, logger=logger) # initialize RLB SAC model

    if path.exists(model_path + "_q1"):
        rlb.load(model_path)
        explore_steps = 10
        logger.info(">> found trained model, initialize explore steps as 10")

    active_as, feature_as, _ = state 

    rlb_state = feature_as[active_as][:, feature_idx].reshape(1, -1) # w/ size (#server*#feature)
    
    # initialize 
    for step in range(max_steps):
        # if step % 10 == 0:
        #     print("rlb_state ({}): {}".format(rlb_state.shape, rlb_state))
        if step > explore_steps:
            action = rlb.get_action(active_as, rlb_state, random_action=False)
        else:
            action = rlb.get_action(active_as, rlb_state, random_action=True)

        # update SAC
        if args.train:
            rlb.update_model(batch_size)

            if step % save_cycle == 0:
                logger.info(">> save model")
                rlb.save(model_path)

        # take next step
        state, reward, _, info = lbenv.step(action)

        active_as, feature_as, _ = state 
        # w/ size (#batch, #feature)
        next_rlb_state = feature_as[active_as][:, feature_idx].reshape(1, -1)

        # push each server's trajectory into replay buffer
        rlb.push_replay_buffer(reward, next_rlb_state)

        rlb_state = next_rlb_state

        # render
        if frame_idx % render_cycle == 0:
            lbenv.render()
