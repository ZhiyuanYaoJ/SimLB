#--- Import ---#
import logging
import sys
import time
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
import socket

#--- MACROS ---#

# Max number of application servers
MAX_N_AS = sm.GLOBAL_CONF["global"]["SHM_N_BIN"]
device = 'cpu'
HOST = None               # Symbolic name meaning all available interfaces
PORT = 50009              # Arbitrary non-privileged port
IP_FMT = '10.0.1.{}'
CENTRAL_SF = 253

#--- Initialization ---#


def get_socket(n_agents, agent_id, timeout=2, logger=None):
    if agent_id == 0:
        # centralized training agent
        S = []
        for i in range(1, n_agents):
            host = IP_FMT.format(CENTRAL_SF-i)
            while True:                
                for res in socket.getaddrinfo(host, PORT, socket.AF_UNSPEC, socket.SOCK_STREAM):
                    s = None
                    af, socktype, proto, canonname, sa = res
                    try:
                        s = socket.socket(af, socktype, proto)
                    except OSError as msg:
                        s = None
                        continue
                    try:
                        s.connect(sa)
                    except OSError as msg:
                        s.close()
                        s = None
                        continue
                    break
                if s is None:
                    if logger:
                        logger.info('could not open socket, retry')
                    time.sleep(.1)
                    # sys.exit(1)
                else:
                    if logger:
                        logger.info('socket created!')
                    break
            S.append(s)
        return S
    else:
        # distributed agent - tcp socket
        s = None
        while True:
            for res in socket.getaddrinfo(HOST, PORT, socket.AF_UNSPEC,
                                        socket.SOCK_STREAM, 0, socket.AI_PASSIVE):
                af, socktype, proto, canonname, sa = res
                try:
                    s = socket.socket(af, socktype, proto)
                except OSError as msg:
                    s = None
                    continue
                try:
                    s.bind(sa)
                    s.listen(1)
                except OSError as msg:
                    s.close()
                    s = None
                    continue
                break
            if s is None:
                if logger:
                    logger.info('could not open socket, retry')
                time.sleep(.1)
                # sys.exit(1)
            else:
                if logger:
                    logger.info('socket created!')                
                break

        conn, _ = s.accept()

        conn.settimeout(timeout)
        # # get sftp socket
        # private_key = "~/.ssh/id_rsa"
        # sftp = pysftp.Connection(
        #     host=IP_FMT.format(CENTRAL_SF), username="cisco", private_key=private_key)
        # # change directory on remote server
        # sftp.chdir('/home/cisco/rl')

        return conn


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

    def push(self, hidden_in, state, action, last_action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            hidden_in, state, action, last_action, reward, next_state)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst = [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        min_seq_len = float('inf')
        for sample in batch:
            h_in, state, action, last_action, reward, next_state = sample
            min_seq_len = min(len(state), min_seq_len)
            # h_in: (1, batch_size=1, n_agents, hidden_size)
            hi_lst.append(h_in)
        hi_lst = torch.cat(hi_lst, dim=-3).detach()  # cat along the batch dim

        # strip sequence length
        for sample in batch:
            h_in, state, action, last_action, reward, next_state = sample
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

        return hi_lst, s_lst, a_lst, la_lst, r_lst, ns_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

    def dump_buffer(self):

        # Saving the objects:
        with open(self.save2file, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self.buffer, self.position], f)


class RNNAgent(nn.Module):
    '''
    @brief:
        evaluate Q value given a state and the action
    '''

    def __init__(self, num_inputs, num_heads, num_actions, hidden_size):
        super(RNNAgent, self).__init__()

        self.num_inputs = num_inputs
        self.num_heads = num_heads
        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_inputs+num_heads*num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.rnn = nn.GRU(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, num_heads*num_actions)

    def forward(self, state, action, hidden_in):
        '''
        @params:
            state: [#batch, #sequence, #agent, #n_feature]
            action: [#batch, #sequence, #agent, num_heads]
            hidden: [#batch, 1, #agent, num_heads]
        @return:
            qs: [#batch, #sequence, #agent, num_heads, num_actions]
        '''
        #  to [#sequence, #batch, #agent, #n_feature]
        # print(state.shape)
        bs, seq_len, n_agents, _ = state.shape
        state = state.permute(1, 0, 2, 3)
        action = action.permute(1, 0, 2, 3)
        action = F.one_hot(action, num_classes=self.num_actions).type(torch.FloatTensor)
        # [#batch, #sequence, #agent, num_heads*num_actions]
        action = action.view(seq_len, bs, n_agents, -1)
        hidden_in = hidden_in.view(1, bs*n_agents, -1)

        x = torch.cat([state, action], -1)  # the dim 0 is number of samples
        # change x to [#sequence, #batch*#agent, -1] to meet rnn's input requirement
        x = x.view(seq_len, bs*n_agents, -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x,  hidden = self.rnn(x, hidden_in)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)  # [#sequence, #batch, #agents, #heads*#actions]
        # [#sequence, #batch, #agent, #head * #action]
        x = x.view(seq_len, bs, n_agents, self.num_heads, self.num_actions)
        # categorical over the discretized actions
        qs = F.softmax(x, dim=-1)
        qs = qs.permute(1, 0, 2, 3, 4)  # permute back

        return qs, hidden

    def get_action(self, state, last_action, hidden_in, deterministic=False):
        '''
        @brief:
            for each distributed agent, generate action for one step given input data
        @params:
            state: [#batch, #feature*num_heads]
            action: [#batch, num_heads]
        '''
        state = torch.FloatTensor(state).unsqueeze(
            0).unsqueeze(-2).to(device)  # add #sequence and #agent dim
        last_action = torch.LongTensor(
            last_action).unsqueeze(0).unsqueeze(-2).to(device)  # add #sequence and #agent dim
        hidden_in = torch.FloatTensor(
            hidden_in).unsqueeze(-2)  # add #agent dim

        agent_outs, hidden_out = self.forward(state, last_action, hidden_in)
        agent_outs = agent_outs.squeeze(-3).to(device)  # remove #agent dim
        hidden_out = hidden_out.squeeze(-3).to(device)  # remove #agent dim
        dist = Categorical(agent_outs)

        if deterministic:
            action = np.argmax(agent_outs.detach().cpu().numpy(), axis=-1)
        else:
            action = dist.sample().squeeze().detach().cpu().numpy()
        return action, hidden_out


class QMix(nn.Module):
    def __init__(self, state_dim, n_agents, num_heads, embed_dim=64, hypernet_embed=128, abs=True):
        """
        Critic network class for Qmix. Outputs centralized value function predictions given independent q value.
        :param args: (argparse) arguments containing relevant model information.
        """
        super(QMix, self).__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim*n_agents  # features*num_heads
        self.num_heads = num_heads

        self.embed_dim = embed_dim
        self.hypernet_embed = hypernet_embed
        self.abs = abs

        # if getattr(args, "hypernet_layers", 1) == 1:
        #     self.hyper_w_1 = nn.Linear(
        #         self.state_dim, self.embed_dim * self.n_agents)
        #     self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        # elif getattr(args, "hypernet_layers", 1) == 2:
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.hypernet_embed, self.num_heads * self.embed_dim * self.n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(self.hypernet_embed, self.embed_dim))
        # elif getattr(args, "hypernet_layers", 1) > 2:
        #     raise Exception("Sorry >2 hypernet layers is not implemented!")
        # else:
        #     raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(
            self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        """
        Compute actions from the given inputs.
        @params:
            agent_qs: [#batch, #sequence, #agent, #num_heads]
            states: [#batch, #sequence, #agent, #features*num_heads]
        :param agent_qs: q value inputs into network [batch_size, #agent, num_heads]
        :param states: state observation.
        :return q_tot: (torch.Tensor) return q-total .
        """
        bs = agent_qs.size(0)
        # [#batch*#sequence, num_heads*#features*#agents]
        states = states.reshape(-1, self.state_dim)

        # [#batch*#sequence, 1, num_heads*#agents]
        agent_qs = agent_qs.reshape(-1, 1, self.num_heads*self.n_agents)
        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(
            states)  # [#batch*#sequence, num_heads*embed_dim*#agents]
        b1 = self.hyper_b_1(states)  # [#batch*#sequence, embed_dim]
        # [#batch*#sequence, num_heads*#agents, embed_dim]
        w1 = w1.view(-1, self.n_agents*self.num_heads, self.embed_dim)
        # [#batch*#sequence, 1, embed_dim]
        b1 = b1.view(-1, 1, self.embed_dim)
        # [#batch*#sequence, 1, embed_dim]
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(
            states)  # [#batch*#sequence, embed_dim]
        # [#batch*#sequence, embed_dim, 1]
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)  # [#batch*#sequence, 1, 1]
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)  # [#batch, #sequence, 1]

        return q_tot

    def k(self, states):
        bs = states.size(0)
        w1 = torch.abs(self.hyper_w_1(states))
        w_final = torch.abs(self.hyper_w_final(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim*self.num_heads)
        w_final = w_final.view(-1, self.embed_dim*self.num_heads, 1)
        k = torch.bmm(w1, w_final).view(bs, -1, self.n_agents)
        k = k / torch.sum(k, dim=2, keepdim=True)
        return k

    def b(self, states):
        bs = states.size(0)
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim*self.num_heads, 1)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim*self.num_heads)
        v = self.V(states).view(-1, 1, 1)
        b = torch.bmm(b1, w_final) + v
        return b

#--- SAC Trainer ---#


class QMix_Trainer():
    def __init__(self, replay_buffer, n_agents, state_dim, num_heads, action_dim, hidden_dim, hypernet_dim, lr=0.001, logger=None):
        self.replay_buffer = replay_buffer

        # if DEBUG:
        #     print("state_dim {}, action_dim {}, hidden_dim {}".format(
        #         state_dim, action_dim, hidden_dim))

        self.agent = RNNAgent(state_dim, num_heads,
                              action_dim, hidden_dim).to(device)
        self.target_agent = RNNAgent(
            state_dim, num_heads, action_dim, hidden_dim).to(device)

        self.mixer = QMix(state_dim, n_agents, num_heads,
                          hidden_dim, hypernet_dim).to(device)
        self.target_mixer = QMix(state_dim, n_agents, num_heads,
                                 hidden_dim, hypernet_dim).to(device)

        self._update_targets()

        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(
            list(self.agent.parameters())+list(self.mixer.parameters()), lr=lr)

        self.logger = logger

    def update(self, batch_size):
        hidden_in, state, action, last_action, reward, next_state = self.replay_buffer.sample(
            batch_size)

        # [#batch, sequence, #agents, #features*num_heads]
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        # [#batch, sequence, #agents, #num_heads]
        action = torch.LongTensor(action).to(device)
        last_action = torch.LongTensor(last_action).to(device)
        # reward is scalar, add 1 dim to be [reward] at the same dim
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(device)

        # [#batch, #sequence, #agent, num_heads, num_actions]
        agent_outs, _ = self.agent(state, last_action, hidden_in)

        # [#batch, #sequence, #n_agent, num_heads]
        chosen_action_qvals = torch.gather(
            agent_outs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
        qtot = self.mixer(chosen_action_qvals, state)

        # target q
        target_agent_outs, _ = self.target_agent(next_state, action, hidden_in)
        # [#batch, #sequence, #agents, num_heads]
        target_max_qvals = target_agent_outs.max(dim=-1, keepdim=True)[0]
        target_qtot = self.target_mixer(target_max_qvals, next_state)

        targets = self._build_td_lambda_targets(reward, target_qtot)

        loss = self.criterion(qtot, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _build_td_lambda_targets(self, rewards, target_qs, gamma=0.99, td_lambda=0.6):
        '''
        @params:
            target_qs: [#batch, #sequence, 1]
        '''
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = target_qs[:, -1]
        # backwards recursive update of the "forward view"
        for t in range(ret.shape[1] - 2, -1, -1):
            ret[:, t] = td_lambda * gamma * ret[:, t+1] + \
                (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t+1])
        return ret

    def _update_targets(self):
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_agent.parameters(), self.agent.parameters()):
            target_param.data.copy_(param.data)

    def save_model(self, path):
        torch.save(self.agent.state_dict(), path+'_agent')
        torch.save(self.mixer.state_dict(), path+'_mixer')

    def load_model(self, path):
        self.agent.load_state_dict(torch.load(path+'_agent'))
        self.mixer.load_state_dict(torch.load(path+'_mixer'))

        self.agent.eval()
        self.mixer.eval()

        self._update_targets()



class RLB_QMix():

    def __init__(
        self,
        lbid,
        n_agents,
        state_dim,
        hidden_dim,
        action_dim,
        head_dim,
        feature_idx,
        hypernet_dim=128,
        replay_buffer_size=3000,
        deterministic=False,
        model_path='rl/rlb_qmix_discrete',
        render_cycle=2,
        sync=True,
        batch_size=16,
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
        self.lbid = lbid
        self.n_agents = n_agents
        # set parameters
        self.logger = logger
        self.replay_buffer = ReplayBufferGRU(
            replay_buffer_size, model_path+'_replay', logger=logger)
        self.action_dim = action_dim
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.feature_idx = feature_idx
        self.deterministic = deterministic
        self.sync = sync  # whether the agents are synchronized or not
        self.render_cycle = render_cycle
        self.train_batch_size = batch_size
        self.model_path = model_path
        self.save2file_fmt = 'agent{}_env.pkl'
        self.t0 = time.time() # for debug

        # initialize channels
        self.tcp_channel = get_socket(n_agents, lbid, logger=logger)

        # initialize model
        self.learner = QMix_Trainer(self.replay_buffer, n_agents, state_dim,
                                    head_dim, action_dim, hidden_dim, hypernet_dim, logger=logger)

        if path.exists(self.model_path + "_agent"):
            self.load(self.model_path)
            self.logger.info(
                ">> found trained model, initialize explore steps as 10")


        self.episode_state = []
        self.episode_action = []
        self.episode_last_action = []
        self.episode_reward_fields = []
        self.episode_next_state = []

        if lbid == 0:
            self.summarize_episode = self.summarize_central
            if self.sync:
                self.run_loop = self.sync_loop_central
            else:
                raise NotImplementedError
        else:
            self.summarize_episode = self.summarize_distrib
            if self.sync:
                self.run_loop = self.sync_loop_distrib
            else:
                raise NotImplementedError

    def save(self, model_path):
        self.learner.save_model(model_path)

    def load(self, model_path):
        self.learner.load_model(model_path)

    def get_action(self, feature_as, last_action):
        '''
        @return:
            action: w/ shape [#active_as]
        '''

        action, self.hidden_state = self.learner.agent.get_action(
            feature_as.reshape(1, -1), last_action.reshape(1, -1), self.hidden_state, deterministic=self.deterministic)

        return action.reshape(-1)

    def summarize_central(self):
        episode_state_all = [self.episode_state]
        episode_action_all = [self.episode_action]
        episode_last_action_all = [self.episode_last_action]
        episode_reward_fields_all = [self.episode_reward_fields]
        episode_next_state_all = [self.episode_next_state]
        seq_len = len(self.episode_state)
        # check from all tcp channels that other agents are done       
        for i, s in enumerate(self.tcp_channel):
            data_recv = s.recv(42)
            assert data_recv.decode('utf-8') == 'done'
            filename = path.join(
                '/home/cisco/rl', self.save2file_fmt.format(i+1))
            with open(filename, 'rb') as f:
                episode_state, episode_action, episode_last_action, episode_reward_fields, episode_next_state = pickle.load(f)
            cmd = 'rm {}'.format(filename)
            subprocess_cmd(cmd)
            episode_state_all.append(episode_state)
            episode_action_all.append(episode_action)
            episode_last_action_all.append(episode_last_action)
            episode_reward_fields_all.append(episode_reward_fields)
            episode_next_state_all.append(episode_next_state)
            seq_len_i = len(episode_action)
            if seq_len > seq_len_i:
                self.logger.info("WARNING: inconsistent seq_len (prev: {} - new: {})".format(seq_len, seq_len_i))
                # choose the shorter one
                seq_len = min(seq_len, seq_len_i)
        
        if seq_len > len(self.episode_state)/2:
            # check sequence length is the same
            episode_state_all = np.array([values[:seq_len] for values in episode_state_all]).transpose(1, 0, 2) # [#agents, #sequence, #features*#heads] -> [#sequence, #agents, #features*#heads]
            episode_action_all = np.array([values[:seq_len] for values in episode_action_all]).transpose(1, 0, 2) # [#agents, #sequence, #heads] -> [#sequence, #agents, #heads]
            episode_last_action_all = np.array([values[:seq_len] for values in episode_last_action_all]).transpose(1, 0, 2) # [#agents, #sequence, #heads] -> [#sequence, #agents, #heads]
            episode_reward_fields_all = np.array([values[:seq_len] for values in episode_reward_fields_all]).transpose(
                1, 0, 2).mean(axis=1)  # [#agents, #sequence, #heads] -> [#sequence, #agents, #heads] -> [#sequence, #heads]
            episode_next_state_all = np.array([values[:seq_len] for values in episode_next_state_all]).transpose(
                1, 0, 2)  # [#agents, #sequence, #features*#heads] -> [#sequence, #agents, #features*#heads]
        
            episode_rewards_all = [calcul_fair_bossaer(reward_fields) for reward_fields in episode_reward_fields_all]

            # push to replay buffer
            self.push_replay_buffer(self.ini_hidden_in, episode_state_all, episode_action_all,
                                    episode_last_action_all, episode_rewards_all, episode_next_state_all)
        else:
            self.logger.info("weird seq info, skip push_replay_buffer")
        self.logger.info("current replay buffer size {}".format(self.replay_buffer.get_length()))

        # update network if possible
        self.update_model(self.train_batch_size)


    def summarize_distrib(self):
        # store episode information and send file via sftp
        filename = path.join('/home/cisco/rl/', self.save2file_fmt.format(self.lbid))
        with open(filename, 'wb') as f:
            pickle.dump([self.episode_state, self.episode_action, self.episode_last_action,
                        self.episode_reward_fields, self.episode_next_state], f)
        cmd = "scp -i /home/cisco/.ssh/id_rsa -oStrictHostKeyChecking=no {} cisco@{}:/home/cisco/rl/".format(filename,
            IP_FMT.format(CENTRAL_SF))
        self.logger.info(subprocess_cmd(cmd))

        self.tcp_channel.send("done".encode('utf-8'))


    def sync_loop_central(self, env, active_as, state, last_action):
        while True:
            action = self.get_action(state, last_action)
            # tell all the other agents to take action
            for s in self.tcp_channel:
                s.send('action'.encode('utf-8'))
                self.logger.info(
                    "DEBUG: ({:.3f}s) sent - [action]".format(time.time()-self.t0))

            obs, _, _, info = env.step(action, active_as)

            # tell all the other agents to make observation
            for s in self.tcp_channel:
                s.send('observe'.encode('utf-8'))
                self.logger.info(
                    "DEBUG: ({:.3f}s) sent - [observe]".format(time.time()-self.t0))
            
            # w/ size (#feature*num_heads)
            next_state = feature_as[active_as][:, self.feature_idx].reshape(-1)

            self.episode_state.append(state)
            self.episode_action.append(action)
            self.episode_last_action.append(last_action)
            self.episode_reward_fields.append(info['reward_fields'])
            self.episode_next_state.append(next_state)
            
            state = next_state
            last_action = action

            # render
            if frame_idx % self.render_cycle == 0:
                env.render()

            # break the episode
            if path.exists('done'):
                break

    def sync_loop_distrib(self, env, active_as, state, last_action):
        
        while True:
            action = self.get_action(state, last_action)
            try:
                data_recv = self.tcp_channel.recv(42)
                # generate action and take action
                if data_recv.decode('utf-8') == 'action':
                    self.logger.info("DEBUG: ({:.3f}s) received - [action]".format(time.time()-self.t0))
                    env.step_take_action(action, active_as)
            except socket.timeout:
                self.logger.info("DEBUG: ({:.3f}s) TIMEOUT when receiving - [action]".format(time.time()-self.t0))
            
            try:
                data_recv = self.tcp_channel.recv(42)
                # get next observation
                if data_recv.decode('utf-8') == 'observe':
                    self.logger.info(
                        "DEBUG: ({:.3f}s) received - [observe]".format(time.time()-self.t0))

                    obs, _, _, info = env.step_get_next_obs()

                    active_as, feature_as, _ = obs
                    # w/ size (#feature*num_heads)
                    next_state = feature_as[active_as][:, self.feature_idx].reshape(-1)

                    self.episode_state.append(state)
                    self.episode_action.append(action)
                    self.episode_last_action.append(last_action)
                    self.episode_reward_fields.append(info['reward_fields'])
                    self.episode_next_state.append(next_state)

                    state = next_state
                    last_action = action

                    # render
                    if frame_idx % self.render_cycle == 0:
                        env.render()
            except socket.timeout:
                self.logger.info(
                    "DEBUG: ({:.3f}s) TIMEOUT when receiving - [observe]".format(time.time()-self.t0))

            # break the episode
            if path.exists('done'):
                break

    def sample_action(self, deterministic=False):
        if deterministic:
            return np.zeros(self.head_dim)
        else:
            probs = torch.FloatTensor(
                np.ones(self.action_dim)/self.action_dim).to(device)
            dist = Categorical(probs)
            action = dist.sample((self.head_dim,))

            return action.type(torch.FloatTensor).numpy()

    def push_replay_buffer(self, ini_hidden_in, episode_state, episode_action, episode_last_action,
                           episode_reward, episode_next_state):
        '''
        @brief: push arguments into replay buffer
        '''
        self.replay_buffer.push(ini_hidden_in, episode_state, episode_action, episode_last_action,
                                episode_reward, episode_next_state)
        self.replay_buffer.dump_buffer()

    def update_model(self, batch_size, update_iter=25):
        t0 = time.time()
        if self.replay_buffer.get_length() >= batch_size: # DEBUG
            for _ in range(update_iter):
                self.logger.info("update model - iter({})".format(_))
                self.learner.update(batch_size)
            self.logger.info(">> save model")
            self.save(self.model_path)
        self.logger.info("training {} iters with {} batch takes {:.3f}s".format(update_iter, batch_size, time.time()-t0))

    def init_state_action(self, state, action):
        self.state_buffer = state,
        self.action_buffer = action
        self.hidden_state = torch.zeros(
            [1, 1, self.hidden_dim], dtype=torch.float)
        self.ini_hidden_in = torch.zeros(
            [1, self.n_agents, self.hidden_dim], dtype=torch.float)
        if self.lbid == 0:
            self.buffer_other_agent = [
                {"state": state, "action": action} for _ in range(self.n_agents-1)]

    def cleanup(self):
        if self.lbid == 0:
            for s in self.tcp_channel:
                s.close()
        else:
            self.tcp_channel.close()
        cmd = "touch /home/cisco/train_done"
        subprocess_cmd(cmd)

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
batch_size = 12 # number of batches for training
# hidden_dim = 128  # number of hidden units (small)
hidden_dim = 128  # number of hidden units
update_itr = 1  # how many iterations do we update each time
max_steps = 9000  # for dev
render_cycle = 2  # every ${render_cycle} steps, print out once current state
feature_idx = [i for i, f in enumerate(
    sm.FEATURE_AS_ALL) if 'flow_duration' in f or 'n_flow_on' in f]  # feature to be used as input
rewards = []
discrete = True
model_path = 'rl/sac_qmix'
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
    logger.info("number of heads: {}".format(head_dim))

    lbenv = LoadBalanceEnv(args.interval, logger,
                           verbose=args.verbose, gt=args.gt, discrete=discrete)
    state = lbenv.reset()  # state consists of (active_as, feature_as, gt)
    rlb = RLB_QMix(
        lbid,
        n_agents,
        len(feature_idx)*head_dim,
        hidden_dim,
        lbenv.num_actions,
        head_dim,
        feature_idx,
        # hypernet_dim=128, # (small)
        hypernet_dim=128,
        model_path=model_path,
        logger=logger,
        batch_size=batch_size,
        )  # initialize RLB QMIX model

    active_as, feature_as, _ = state
    # w/ size (#server*#feature)
    rlb_state = feature_as[active_as][:, feature_idx].reshape(-1)
    # initialize
    last_action = rlb.sample_action(deterministic=True)
    rlb.init_state_action(rlb_state, last_action)

    rlb.run_loop(lbenv, active_as, rlb_state, last_action)

    if args.train:
        rlb.summarize_episode()

    rlb.cleanup()
    lbenv.cleanup()
    
