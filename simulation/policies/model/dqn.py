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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import struct

#--- MACROS ---#
DEVICE = torch.device("cpu")
REPLAY_BUFFER_SIZE = 3000

#--- Initialization ---#


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

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(message)s',
                        handlers=[
                            logging.FileHandler(filename=filename),
                            logging.StreamHandler(sys.stdout)
                        ])

    # Test
    logger = logging.getLogger(logger_name)
    logger.info('### Init. Logger {} ###'.format(logger_name))
    return logger


#--- Buffer ---#

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
            action: 1d tensor w/ length=action_dim
            reward: scalar
        @dev:
            for now we don't consider gt at all w/ RL, that is gt=None
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (  # (active_as, next_active_as, feature_lb, feature_as, next_feature_lb, next_feature_as, action, reward)
                state[0], next_state[0], state[1], state[2], next_state[1],
                next_state[2], action, reward)
        self.position = int(
            (self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        active_as = [sample[0] for sample in batch]
        next_active_as = [sample[1] for sample in batch]
        batch2stack = [sample[2:] for sample in batch]
        feature_lb, feature_as, next_feature_lb, next_feature_as, action, reward = map(
            np.stack, zip(*batch2stack))

        return [active_as, feature_lb, feature_as], action, reward, [next_active_as, next_feature_lb, next_feature_as]

    def __len__(self):
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

#--- Value Networks ---#


class DQN(nn.Module):
    '''
    @brief:
        a very simple implementation, taking all steps as independent features
    '''

    def __init__(self,
                 n_feature_as,
                 n_feature_lb,
                 action_dim,
                 hidden_size,
                 init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.action_dim = action_dim
        self.n_feature_lb = n_feature_lb
        self.n_feature_as = n_feature_as
        self.state_dim = n_feature_lb + action_dim * n_feature_as

        self.bn_as = nn.BatchNorm1d(n_feature_as)
        self.bn_lb = nn.BatchNorm1d(n_feature_lb - n_feature_as)
        self.linear1 = nn.Linear(self.state_dim + self.action_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        # self.linear2=nn.Linear(hidden_size, hidden_size)
        # self.linear3=nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, self.action_dim)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        '''
        @param:
            state: a tuple consists of:
                active_as: consists of #batch lists of active AS id
                feature_lb: torch.FloatTensor w/ shape [#batch, #n_feature_lb] where the first #n_feature_as cols are averaged across active AS nodes and the last (#n_feature_lb-#n_feature_as) cols are gathered locally on LB node
                feature_as: torch.FloatTensor w/ shape [#batch, action_dim, #n_feature_as]
            action: torch.FloatTensor w/ shape [#batch, action_dim]
        '''
        active_as, feature_lb, feature_as = state
        n_batch = feature_lb.shape[0]
        feature_as_bn_buffer = torch.zeros(n_batch, self.action_dim,
                                           self.n_feature_as).to(DEVICE)

        # pass observations through a batch normalization layer
        # w/ shape [#n_batch, #n_feature_lb-#n_feature_as]
        obs_lb = self.bn_lb(feature_lb[:, self.n_feature_as:])
        obs_as = self.bn_as(
            torch.cat([
                feature_lb[:, :self.n_feature_as],
                torch.cat([
                    feature_as[i, active_as_, :]
                    for i, active_as_ in enumerate(active_as)
                ], 0)
            ], 0)
        )  # w/ shape [#n_batch+sum(#n_active_as_each_batch), #n_feature_as]

        # reshape all features to a single tensor w/ #n_batch rows
        cnt_ = n_batch
        for i, active_as_ in enumerate(active_as):
            n_active_ = len(active_as_)
            feature_as_bn_buffer[i, active_as_] = obs_as[cnt_:cnt_ + n_active_]
            cnt_ += n_active_
        x = torch.cat([
            obs_lb, obs_as[:n_batch],
            feature_as_bn_buffer.reshape(n_batch, -1), action
        ], 1)  # concat all features and actions into #n_batch rows

        x = F.elu(self.linear1(x))
        x = self.ln1(x)
        # x=F.elu(self.linear2(x))
        # x=F.elu(self.linear3(x))
        x = self.linear4(x)
        return x


#--- DQN Trainer ---#


class DQN_Trainer():
    def __init__(self,
                 replay_buffer,
                 n_feature_as,
                 n_feature_lb,
                 hidden_dim,
                 action_range,
                 action_dim,
                 logger=None):
        self.replay_buffer = replay_buffer
        print(DEVICE)
        self.policy_net = DQN(n_feature_as, n_feature_lb, action_dim,
                                        hidden_dim).to(DEVICE)
        self.target_net = DQN(n_feature_as, n_feature_lb, action_dim,
                                        hidden_dim).to(DEVICE)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.criterion = nn.MSELoss()

        lr = 3e-4
        
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                            lr=lr)

    def update(self,
               batch_size,
               reward_scale=10.,
               auto_entropy=True,
               target_entropy=-2,
               gamma=0.99,
               soft_tau=1e-2):
        state, action, reward, next_state = self.replay_buffer.sample(
            batch_size)

        state[1] = torch.FloatTensor(state[1]).to(DEVICE)
        state[2] = torch.FloatTensor(state[2]).to(DEVICE)
        next_state[1] = torch.FloatTensor(next_state[1]).to(DEVICE)
        next_state[2] = torch.FloatTensor(next_state[2]).to(DEVICE)
        action = torch.FloatTensor(action).to(DEVICE)
        # reward is scalar, add 1 dim to be [reward] at the same dim
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(
            state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(
            next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (
            reward.std(dim=0) + 1e-6)

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
        target_q_min = torch.min(
            self.target_soft_q_net1(next_state, new_next_action),
            self.target_soft_q_net2(
                next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + gamma * target_q_min
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1,
                                               target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2,
                                               target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # update Policy
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),
                                          self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(),
                                       self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for target_param, param in zip(self.target_soft_q_net2.parameters(),
                                       self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '_q1')
        torch.save(self.soft_q_net2.state_dict(), path + '_q2')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path + '_q2'))
        self.policy_net.load_state_dict(torch.load(path + '_policy'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()
