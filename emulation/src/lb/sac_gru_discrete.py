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
from torch.distributions import Categorical

import shm_proxy as sm
from env import *
import argparse
import pickle
import subprocess

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


def subprocess_cmd(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    return proc_stdout.decode("utf-8")


class ReplayBufferGRU:
    """ 
    Replay buffer for agent with GRU network additionally storing previous action, 
    initial input hidden state and output hidden state of GRU.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for GRU initialization.

    """

    def __init__(self, capacity, init_filename, logger=None):
        self.save2file = init_filename
        self.capacity = capacity
        if path.exists(init_filename):
            # Getting back the objects:
            with open(init_filename, 'rb') as f:
                self.buffer, self.position = pickle.load(f)
                if logger:
                    logger.info("replay_buffer: {}, {}".format(self.position, len(self.buffer)))
        else:
            self.buffer = []
            self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            hidden_in, hidden_out, state, action, last_action, reward, next_state)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ho_lst = [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        min_seq_len = float('inf')
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state = sample
            min_seq_len = min(len(state), min_seq_len)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ho_lst.append(h_out)
        hi_lst = torch.cat(hi_lst, dim=-2).detach()  # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()

        
        # strip sequence length
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state = sample
            sample_len = len(state)
            start_idx = int((sample_len - min_seq_len)/2)
            end_idx = start_idx+min_seq_len
            s_lst.append(state[start_idx:end_idx])
            a_lst.append(action[start_idx:end_idx])
            la_lst.append(last_action[start_idx:end_idx])
            r_lst.append(reward[start_idx:end_idx])
            ns_lst.append(next_state[start_idx:end_idx])
            # print('sample_len: {} taken {}-{}'.format(sample_len, start_idx, end_idx))
            # print("state.shape: {}".format(np.array(state).shape))
            # print("last_action.shape: {}".format(np.array(last_action).shape))

        # print("s_lst.shape: {}".format(np.array(s_lst).shape))
        # print("a_lst.shape: {}".format(np.array(a_lst).shape))
        # print("la_lst.shape: {}".format(np.array(la_lst).shape))
        # print("r_lst.shape: {}".format(np.array(r_lst).shape))
        # print("ns_lst.shape: {}".format(np.array(ns_lst).shape))
        

        return hi_lst, ho_lst, s_lst, a_lst, la_lst, r_lst, ns_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

    def dump_buffer(self):

        # Saving the objects:
        with open(self.save2file, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self.buffer, self.position], f)


class SoftQNetworkGRU(nn.Module):
    '''
    @brief:
        evaluate Q value given a state and the action
    '''

    def __init__(self, num_inputs, num_heads, hidden_size, init_w=3e-3):
        super(SoftQNetworkGRU, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_heads, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.rnn = nn.GRU(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, hidden_in):
        # [#batch, #sequence, #n_feature] to [#sequence, #batch, #n_feature]
        # print(state.shape)
        state = state.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        x = torch.cat([state, action], -1)  # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x,  hidden = self.rnn(x, hidden_in)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        x = x.permute(1, 0, 2)  # back to same axes as input
        return x, hidden


class PolicyNetworkGRU(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, num_heads, logger=None):
        super(PolicyNetworkGRU, self).__init__()
        self.logger = logger
        self.num_actions = num_actions
        self.num_heads = num_heads

        self.linear1 = nn.Linear(num_inputs+num_heads, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.rnn = nn.GRU(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, num_actions*num_heads)

    def forward(self, state, last_action, hidden_in, softmax_dim=-1):
        '''
        @param:
            state: torch.FloatTensor w/ shape [#batch, #sequence, #n_feature]
        @return:
            probs: [#batch, #sequence, num_heads, num_actions]
        '''
        state = state.permute(1, 0, 2)  # [#sequence, #batch, #n_feature]
        last_action = last_action.permute(1, 0, 2)
        x = torch.cat([state, last_action], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x,  hidden = self.rnn(x, hidden_in)
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.output(x)
        x = x.view(x.shape[0], -1, self.num_heads, self.num_actions) # [#sequence, #batch, #head * #action]
        # categorical over the discretized actions
        probs = F.softmax(x, dim=softmax_dim)
        probs = probs.permute(1, 0, 2, 3)  # permute back

        return probs, hidden

    def evaluate(self, state, last_action, hidden_in, epsilon=1e-6):
        '''
        @brief:
            generate sampled action with state as input wrt the policy network
        '''

        # (batch, num_heads, num_actions)
        probs, hidden_out = self.forward(
            state, last_action, hidden_in, softmax_dim=-1)
        dist = Categorical(probs)
        action = dist.sample()  # (batch, sequence, num_heads)
        print("evaluate: state.shape {}, last_action.shape {}, probs.shape {}, action.shape {}".format(
            state.shape, last_action.shape, probs.shape, action.shape))

        log_probs = dist.log_prob(action)
        log_probs = torch.sum(log_probs, dim=-1).unsqueeze(-1)
        
        return action, log_probs, hidden_out

    def get_action(self, state, last_action, hidden_in, deterministic):
        '''
        @param:
            state: w/ shape [#batch, n_feature_as*num_actions]
            action: w/ shape [#batch, num_actions]
        @return:

            action: w/ shape [num_actions]
        '''

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        last_action = torch.FloatTensor(last_action).unsqueeze(0).to(device)

        probs, hidden_out = self.forward(state, last_action, hidden_in)
        dist = Categorical(probs)

        if deterministic:
            action = np.argmax(probs.detach().cpu().numpy(), axis=-1)
        else:
            action = dist.sample().squeeze().detach().cpu().numpy()
        return action, hidden_out

#--- SAC Trainer ---#


class SAC_Trainer():
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim, head_dim, logger=None):
        self.replay_buffer = replay_buffer

        # if DEBUG:
        #     print("state_dim {}, action_dim {}, hidden_dim {}".format(
        #         state_dim, action_dim, hidden_dim))

        self.soft_q_net1 = SoftQNetworkGRU(
            state_dim, head_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetworkGRU(
            state_dim, head_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetworkGRU(
            state_dim, head_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetworkGRU(
            state_dim, head_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetworkGRU(
            state_dim, action_dim, hidden_dim, head_dim, logger=logger).to(device)
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
        hidden_in, hidden_out, state, action, last_action, reward, next_state = self.replay_buffer.sample(
            batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        last_action = torch.FloatTensor(last_action).to(device)
        # reward is scalar, add 1 dim to be [reward] at the same dim
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(device)

        predicted_q_value1, _ = self.soft_q_net1(state, action, hidden_in)
        predicted_q_value2, _ = self.soft_q_net2(state, action, hidden_in)
        new_action, log_prob, _ = self.policy_net.evaluate(
            state, last_action, hidden_in)
        new_next_action, next_log_prob, _ = self.policy_net.evaluate(
            next_state, action, hidden_out)
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
        predict_target_q1, _ = self.target_soft_q_net1(
            next_state, new_next_action.type(torch.FloatTensor), hidden_out)
        predict_target_q2, _ = self.target_soft_q_net2(
            next_state, new_next_action.type(torch.FloatTensor), hidden_out)
        target_q_min = torch.min(
            predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
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
        predict_q1, _ = self.soft_q_net1(state, new_action.type(torch.FloatTensor), hidden_in)
        predict_q2, _ = self.soft_q_net2(
            state, new_action.type(torch.FloatTensor), hidden_in)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        print("predict_q1.shape {} predict_q2.shape {}".format(predict_q1.shape, predict_q2.shape))
        
        print("log_prob.shape {} predicted_new_q_value.shape {}".format(log_prob.shape, predicted_new_q_value.shape))
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
        head_dim,
        replay_buffer_size=3000,
        auto_entropy=True,
        deterministic=False,
        model_path='rl/rlb_sac',
        logger=None
    ):
        '''
        @brief:
            This class generate actions (weights) based on features gathered by reservoir sampling
        @param:
            feature_name: feature keyword that we use to calculate weights (choose among sm.FEATURE_AS_ALL)
            map_func: map function e.g. reciprocal or negative
            logger: logging info
        '''
        self.n_as = sm.GLOBAL_CONF['global']['SHM_N_BIN']  # n_as = num_actions
        # set parameters
        self.logger = logger
        self.replay_buffer = ReplayBufferGRU(
            replay_buffer_size, model_path+'_replay', logger=logger)
        self.action_dim = action_dim
        self.head_dim = head_dim
        self.auto_entropy = auto_entropy
        self.deterministic = deterministic
        self.model_path = model_path

        # initialize model
        self.sac_trainer = SAC_Trainer(
            self.replay_buffer, state_dim, action_dim, hidden_dim, head_dim, logger)

    def save(self, model_path):
        self.sac_trainer.save_model(model_path)

    def load(self, model_path):
        self.sac_trainer.load_model(model_path)

    def get_action(self, feature_as, last_action, hidden_in):
        '''
        @return:
            action: w/ shape [#num_heads]
        '''

        action, hidden_out = self.sac_trainer.policy_net.get_action(
            feature_as.reshape(1, -1), last_action.reshape(1, -1), hidden_in, deterministic=self.deterministic)

        return action.reshape(-1), hidden_out

    def sample_action(self):
        probs = torch.FloatTensor(
            np.ones(self.action_dim)/self.action_dim).to(device)
        dist = Categorical(probs)
        action = dist.sample((self.head_dim,))

        return action.type(torch.FloatTensor).numpy()

    def push_replay_buffer(self, ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action,
                           episode_reward, episode_next_state):
        '''
        @brief: push arguments into replay buffer
        '''
        self.replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action,
                                episode_reward, episode_next_state)
        self.replay_buffer.dump_buffer()

    def update_model(self, batch_size, update_iter=5):
        if self.replay_buffer.get_length() >= batch_size:
            for _ in range(update_iter):
                self.logger.info("update model - iter({})".format(_))
                self.sac_trainer.update(
                    batch_size,
                    reward_scale=1,
                    auto_entropy=self.auto_entropy,
                    target_entropy=-1.,
                )

            self.logger.info(">> save model")
            self.save(self.model_path)


#--- Arguments ---#

parser = argparse.ArgumentParser(
    description='Load Balance Environment w/ Openai Gym APIs.')

parser.add_argument('-v', action='store_true',
                    default=False,
                    dest='verbose',
                    help='Set verbose mode and print out all info')

parser.add_argument('-d', action='store_true',
                    default=False,
                    dest='dev',
                    help='Set dev mode and test offline without opening shared memory file')

parser.add_argument('-m', action='store_true',
                    default='alias',
                    dest='method',
                    help='Set method to encode action [\'alias\' for weighted-sampling, \'score\' for deterministic evaluation]')

parser.add_argument('-i', action='store',
                    default=0.25,
                    dest='interval',
                    help='Set sleep interval in env.step() for action to take effect')

parser.add_argument('-g', action='store_true',
                    default=False,
                    dest='gt',
                    help='Set if collect ground truth')

parser.add_argument('-t', action='store_false',
                    default=True,
                    dest='train',
                    help='Set to False if no need to train')

parser.add_argument('--version', action='version',
                    version='%(prog)s 1.0')

#--- Macros ---#
frame_idx = 0  # number of iterations
batch_size = 12  # number of batches for training
# hidden_dim = 128  # number of hidden units (small)
hidden_dim = 512  # number of hidden units
update_itr = 1  # how many iterations do we update each time
max_steps = 9000  # for dev
render_cycle = 2  # every ${render_cycle} steps, print out once current state
feature_idx = [i for i, f in enumerate(
    sm.FEATURE_AS_ALL) if 'flow_duration' in f or 'n_flow_on' in f]  # feature to be used as input
rewards = []
discrete = True
model_path = 'rl/sac_gru'
if discrete:
    model_path += '_discrete'
# DEBUG = True

if __name__ == '__main__':
    logger = init_logger("log/logger.log", "heuristic-logger")

    args = parser.parse_args()

    lines = []
    with open('/home/cisco/topo', 'r') as f:
        lines = [line.rstrip('\n') for line in f]
    lbid, n_agents, head_dim = lines[0].split('/')
    lbid, n_agents, head_dim = int(lbid), int(n_agents), int(head_dim)


    lbenv = LoadBalanceEnv(args.interval, logger,
                           verbose=args.verbose, gt=args.gt, discrete=discrete)
    state = lbenv.reset()  # state consists of (active_as, feature_as, gt)
    rlb = RLB_SAC(len(feature_idx)*head_dim, hidden_dim, lbenv.num_actions,
                  head_dim, model_path=model_path, logger=logger)  # initialize RLB SAC model

    if path.exists(model_path + "_q1"):
        rlb.load(model_path)
        logger.info(">> found trained model, initialize explore steps as 10")

    active_as, feature_as, _ = state

    rlb_state = feature_as[active_as][:, feature_idx].reshape(-1)  # w/ size (#server*#feature)

    # initialize
    hidden_out = torch.zeros([1, 1, hidden_dim], dtype=torch.float)
    last_action = rlb.sample_action()
    episode_state = []
    episode_action = []
    episode_last_action = []
    episode_reward = []
    episode_next_state = []
    for step in range(max_steps):
        # if step % 10 == 0:
        #     print("rlb_state ({}): {}".format(rlb_state.shape, rlb_state))
        hidden_in = hidden_out
        action, hidden_out = rlb.get_action(
            rlb_state, last_action, hidden_in)

        # take next step
        state, reward, _, info = lbenv.step(action, active_as)

        active_as, feature_as, _ = state
        # w/ size (#batch, #feature)
        next_rlb_state = feature_as[active_as][:, feature_idx].reshape(-1)


        if step == 0:
            ini_hidden_in = hidden_in
            ini_hidden_out = hidden_out
        episode_state.append(rlb_state)
        episode_action.append(action)
        # print("last_action (init): {}".format(last_action))

        episode_last_action.append(last_action)
        episode_reward.append(reward)
        episode_next_state.append(next_rlb_state)

        rlb_state = next_rlb_state
        last_action = action
        # print("episode_state shape {}".format(np.array(episode_state).shape))
        # print("episode_action shape {}".format(np.array(episode_action).shape))
        # print("episode_last_action {}".format(np.array(episode_last_action)))
        # print("episode_last_action shape {}".format(np.array(episode_last_action).shape))

        # render
        if frame_idx % render_cycle == 0:
            lbenv.render()

        # break the episode
        if path.exists('done'):
            break


    # update SAC
    if args.train:
        rlb.push_replay_buffer(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action,
                                episode_reward, episode_next_state)
        logger.info(">> number of episodes: {} (each shape {})".format(
            len(episode_state), episode_state[0].shape))

        rlb.update_model(batch_size)

        # mark train is over
        cmd = "touch /home/cisco/train_done"
        subprocess_cmd(cmd)

    lbenv.cleanup()
