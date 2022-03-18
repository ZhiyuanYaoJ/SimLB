#--- Import ---#
import math
import time
import random
import numpy as np
import gym
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import shm_manager as sm
import argparse
import struct
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#--- MACROS ---#

ACTION_TYPE_OPTIONS = ['score', 'alias']
# Regret version, e.g. -(weighted_average(mean(FCT))), clip value to avoid dramatic gradient
REGRET_RANGE = (-5, 0)
# Reward version, e.g. fairness, clip value to avoid dramatic gradient
REWARD_RANGE = (0, 1)
SEQ_LEN = 64  # How many time steps do we look back into the history
# Maximum value in the feature range, so as to avoid seeing NaNs w/ np.inf
MAX_FEATURE_VALUE = np.power(2., 16)
TEMPORAL_RANGE = (0., MAX_FEATURE_VALUE)  # Temporal feature range
COUNTER_RANGE = (0., MAX_FEATURE_VALUE)  # Counter feature range
WIN_RANGE = (0, 65535)  # Window size feature range
DWIN_RANGE = (-32768, 32768)  # 1st derivative of window size feature range
BYTE_PACKET_RANGE = (0, 1500)  # Byte transmitted per packet feature range
# Byte transmitted per flow feature range
BYTE_FLOW_RANGE = (0, MAX_FEATURE_VALUE)
FEATURES2USE = {  # Actual features to be used and their corresponding range
    ##--- Temporal ---##
    "fct_avg": TEMPORAL_RANGE,
    "fct_std": TEMPORAL_RANGE,
    "iat_ppf_avg": TEMPORAL_RANGE,
    "iat_ppf_std": TEMPORAL_RANGE,
    "lat_synack_avg": TEMPORAL_RANGE,
    "lat_synack_std": TEMPORAL_RANGE,
    "pt_1st_avg": TEMPORAL_RANGE,
    "pt_1st_std": TEMPORAL_RANGE,
    "pt_gen_avg": TEMPORAL_RANGE,
    "pt_gen_std": TEMPORAL_RANGE,
    "iat_p_avg": TEMPORAL_RANGE,
    "iat_p_std": TEMPORAL_RANGE,
    "iat_f_avg": TEMPORAL_RANGE,
    "iat_f_std": TEMPORAL_RANGE,
    ##--- Counter ---##
    "n_flow": COUNTER_RANGE,
    "n_packet": COUNTER_RANGE,
    "n_fct": COUNTER_RANGE,
    "n_lat_synack": COUNTER_RANGE,
    "n_pt_1st": COUNTER_RANGE,
    "n_pt_gen": COUNTER_RANGE,
    "n_norm_ack": COUNTER_RANGE,
    "n_rtr": COUNTER_RANGE,
    "n_dpk": COUNTER_RANGE,
    ##--- Networking ---##
    "win_avg": WIN_RANGE,
    "win_std": WIN_RANGE,
    "dwin_avg": DWIN_RANGE,
    "dwin_std": DWIN_RANGE,
    "byte_p_avg": BYTE_PACKET_RANGE,
    "byte_p_std": BYTE_PACKET_RANGE,
    ##--- Reservoir ---##
    "res_byte_f_avg": BYTE_FLOW_RANGE,
    "res_byte_f_90": BYTE_FLOW_RANGE,
    "res_byte_f_avg_decay": BYTE_FLOW_RANGE,
    "res_byte_f_90_decay": BYTE_FLOW_RANGE,
    "res_fct_avg": TEMPORAL_RANGE,
    "res_fct_90": TEMPORAL_RANGE,
    "res_fct_avg_decay": TEMPORAL_RANGE,
    "res_fct_90_decay": TEMPORAL_RANGE,
    "res_flow_duration_avg": TEMPORAL_RANGE,
    "res_flow_duration_90": TEMPORAL_RANGE,
    "res_flow_duration_avg_decay": TEMPORAL_RANGE,
    "res_flow_duration_90_decay": TEMPORAL_RANGE,
}
device = 'cpu'
    
#--- Initialization ---#


def linear_weights_init(m):
    if isinstance(m, nn.Linear):
        stdv = 1. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)

#--- Buffer ---#


class ReplayBuffer(object):
    '''
    @brief:
        A simple implementation of ring buffer storing tuple (observation, action, rewards, next_observation)
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, obs, action, reward, next_obs):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, action, reward, next_obs)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs = map(np.stack, zip(*batch))
        return obs, action, reward, next_obs

    def __len__(self):
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


class ReplayBufferLSTM(ReplayBuffer):
    '''
    @brief:
        Replay ring buffer for agent w/ LSTM network, storing previous action, initial input hidden and output hidden states of LSTM in addition.
        Each sample contains a list of steps (aka episode) instead of a single step.
        'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.
    '''

    def __init__(self, capacity):
        super(ReplayBufferLSTM, self).__init__()

    def push(self, hidden_in, hidden_out, obs, action, last_action, reward, next_obs):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            hidden_in, hidden_out, obs, action, last_action, reward, next_obs)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        o_lst, a_lst, la_lst, r_lst, no_lst, hi_lst, ci_lst, ho_lst, co_lst = [
        ], [], [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            (h_in, c_in), (h_out,
                           c_out), obs, action, last_action, reward, next_obs = sample
            o_lst.append(obs)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            no_lst.append(next_obs)
            hi_lst.append(h_in)  # h_in.shape: (1, batch_size=1, hidden_size)
            ci_lst.append(c_in)
            ho_lst.append(h_out)
            co_lst.append(c_out)
        hi_lst = torch.cat(hi_lst, dim=-2).detach()  # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()
        ci_lst = torch.cat(ci_lst, dim=-2).detach()
        ci_lst = torch.cat(co_lst, dim=-2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)

        return hidden_in, hidden_out, o_lst, a_lst, la_lst, r_lst, no_lst


class ReplayBufferGRU(ReplayBuffer):
    '''
    @brief:
        Replay ring buffer for agent w/ GRU network, storing previous action, initial input hidden and output hidden states of GRU in addition.
        Each sample contains a list of steps (aka episode) instead of a single step.
        'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for GRU initialization.
    '''

    def __init__(self, capacity):
        super(ReplayBufferGRU, self).__init__()

    def push(self, hidden_in, hidden_out, obs, action, last_action, reward, next_obs):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            hidden_in, hidden_out, obs, action, last_action, reward, next_obs)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        o_lst, a_lst, la_lst, r_lst, no_lst, hi_lst, ho_lst = [], [], [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            h_in, h_out, obs, action, last_action, reward, next_obs = sample
            o_lst.append(obs)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            no_lst.append(next_obs)
            hi_lst.append(h_in)  # h_in.shape: (1, batch_size=1, hidden_size)
            ho_lst.append(h_out)
        hi_lst = torch.cat(hi_lst, dim=-2).detach()  # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()

        return hi_lst, ho_lst, o_lst, a_lst, la_lst, r_lst, no_lst

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

#--- Value Networks ---#


class ValueNetworkBase(nn.Module):
    '''
    @brief: Base network class for value function approximation
    '''

    def __init__(self, state_space, activation):
        super(ValueNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state
            pass
        self.activation = activation

    def forward(self):
        raise NotImplementedError


class QNetworkBase(ValueNetworkBase):
    def __init__(self, state_space, action_space, activation):
        super().__init__(state_space, activation)
        self._action_space = action_space
        self._action_shape = action_space.shape
        self._action_dim = self._action_shape[0]


class QNetwork(QNetworkBase):
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self.linear1 = nn.Linear(
            self._state_dim + self._action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        self.linear4.apply(linear_weights_init)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # dim 0 = number of samples
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x


class QNetworkLSTM(QNetworkBase):
    '''
    @brief:
        Q network w/ LSTM structure.
        It follows two-branch structure as in paper: Sim-to-Real Transfer of Robotic Control w/ Dynamics Randomization.
        One branch for (state, action), the other for (state, last_action)
    '''

    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(
            self._state_dim + self._action_dim, hidden_dim)  # branch 1
        self.linear2 = nn.Linear(
            self._state_dim + self._action_dim, hidden_dim)  # branch 2
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        self.linear4.apply(linear_weights_init)  # weights initialization

    def forward(self, state, action, last_action, hidden_in):
        '''
        @params_shape:
            state:  (batch_size, sequence_length, state_dim)
            output: (batch_size, sequence_length)
        @note:
            for pytorch lstm, needs to be permuted as: (sequence_length, batch_size, state_dim)
        '''
        # permute
        state = state.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)
        # branch 1
        fc_branch = torch.cat([state, action], -1)
        fc_branch = self.activation(self.linear1(fc_branch))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = self.activation(self.linear2(lstm_branch))
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)
        # merged branch
        merged_branch = torch.cat([fc_branch, lstm_branch], -1)

        x = self.activation(self.linear3(merged_branch))
        x = self.linear4(x)

        # permute back
        x = x.permute(1, 0, 2)
        return x, lstm_hidden


class QNetworkLSTM1(QNetworkBase):
    '''
    @brief:
        Q network w/ LSTM structure.
        It follows single-branch structure as in paper: Memory-based control w/ RNN.
        Only one branch for (state, action, last_action).
    '''

    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(
            self._state_dim + 2 * self._action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear3.apply(linear_weights_init)  # weights initialization

    def forward(self, state, action, last_action, hidden_in):
        '''
        @params_shape:
            state:  (batch_size, sequence_length, state_dim)
            output: (batch_size, sequence_length)
        @note:
            for pytorch lstm, needs to be permuted as: (sequence_length, batch_size, state_dim)
        '''
        # permute
        state = state.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)
        # single branch
        x = torch.cat([state, action, last_action], -1)
        x = self.activation(self.linear1(x))
        x, lstm_hidden = self.lstm1(x, hidden_in)
        x = self.activation(self.linear2(merged_branch))
        x = self.linear3(x)
        # permute back
        x = x.permute(1, 0, 2)
        return x, lstm_hidden


class QNetworkGRU(QNetworkBase):
    '''
    @brief:
        Q network w/ GRU structure.
        It follows two-branch structure as in paper: Sim-to-Real Transfer of Robotic Control w/ Dynamics Randomization.
        One branch for (state, action), the other for (state, last_action)
    '''

    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(
            self._state_dim + self._action_dim, hidden_dim)  # branch 1
        self.linear2 = nn.Linear(
            self._state_dim + self._action_dim, hidden_dim)  # branch 2
        self.gru1 = nn.GRU(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        self.linear4.apply(linear_weights_init)  # weights initialization

    def forward(self, state, action, last_action, hidden_in):
        '''
        @params_shape:
            state:  (batch_size, sequence_length, state_dim)
            output: (batch_size, sequence_length)
        @note:
            for pytorch gru, needs to be permuted as: (sequence_length, batch_size, state_dim)
        '''
        # permute
        state = state.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)
        # branch 1
        fc_branch = torch.cat([state, action], -1)
        fc_branch = self.activation(self.linear1(fc_branch))
        # branch 2
        gru_branch = torch.cat([state, last_action], -1)
        gru_branch = self.activation(self.linear2(gru_branch))
        gru_branch, gru_hidden = self.gru1(gru_branch, hidden_in)
        # merged branch
        merged_branch = torch.cat([fc_branch, gru_branch], -1)

        x = self.activation(self.linear3(merged_branch))
        x = self.linear4(x)

        # permute back
        x = x.permute(1, 0, 2)
        return x, gru_hidden


class QNetworkGRU1(QNetworkBase):
    '''
    @brief:
        Q network w/ GRU structure.
        It follows single-branch structure as in paper: Memory-based control w/ RNN.
        Only one branch for (state, action, last_action).
    '''

    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(
            self._state_dim + 2 * self._action_dim, hidden_dim)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear3.apply(linear_weights_init)  # weights initialization

    def forward(self, state, action, last_action, hidden_in):
        '''
        @params_shape:
            state:  (batch_size, sequence_length, state_dim)
            output: (batch_size, sequence_length)
        @note:
            for pytorch gru, needs to be permuted as: (sequence_length, batch_size, state_dim)
        '''
        # permute
        state = state.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)
        # single branch
        x = torch.cat([state, action, last_action], -1)
        x = self.activation(self.linear1(x))
        x, gru_hidden = self.gru1(x, hidden_in)
        x = self.activation(self.linear2(merged_branch))
        x = self.linear3(x)
        # permute back
        x = x.permute(1, 0, 2)
        return x, gru_hidden

#--- Environment ---#


class LoadBalanceEnv(gym.Env):
    '''
    @brief:
        An environment for load-aware load-balancing respecting openai gym API.
    @state:
        States are a series of LB local observations that are fetched from shared memory with a fixed frequency.
    @action:
        Action space is a list of softmax weights over all servers. [TBD]
    @reward:
        Reward can be calculated according to flow complete time and/or flow duration gathered in reservoir sampling. [TBD]
    '''
    metadata = {'render.modes': ['cli']}

    def __init__(self, action_type, interval):
        '''
        @brief:
            Initialize interfaces 
        @params:
            action_type: whether the output is score (to be used by po2) or weights (to be used by the alias methods)
            interval: wait a bit to gather reward when taking one step
        '''
        super(LoadBalanceEnv, self).__init__()
        # Initialize action type (use 'score' or 'alias')
        assert action_type in ACTION_TYPE_OPTIONS
        self.action_type = action_type
        # initialize sleep interval
        assert interval > 0
        self.interval = interval
        # Get communication API w/ shared memory
        self.shm_m = sm.Shm_Manager(sm.CONF_FILE)
        # Set reward range
        self.reward_range = REGRET_RANGE
        self.reward_field = 'res_flow_duration_avg_decay'
        self.reward_field_idx = [
            k for k, v in FEATURES2USE.items()].index(self.reward_field)
        # Actions of the format [score (or weight)] * N_AS
        self.action_space = spaces.Box(
            low=np.array([0]*sm.N_AS),
            high=np.array([1]*sm.N_AS),
            dtype=np.float32
        )
        # Set feature mask (select only features that we want to use)
        self.feature_mask = tuple([i for i, v in enumerate(
            self.shm_m.feature_keys) if v in FEATURES2USE.keys()])
        # initialize
        self.gt_sockets = sm.get_sockets(sm.HOST, sm.PORT)
        #
        self.observation_space = spaces.Tuple((
            spaces.Box(
                low=np.repeat(np.array([v[0] for i, v in FEATURES2USE.items()]).reshape(
                    1, -1), SEQ_LEN, axis=0),
                high=np.repeat(np.array([v[1] for i, v in FEATURES2USE.items()]).reshape(
                    1, -1), SEQ_LEN, axis=0),
                dtype=np.float32
            ),
            spaces.Discrete(self.shm_m.shm_n_bin)
        ))

    def reset(self):
        '''
        @brief:
            Reset the state of the environment to an initial state
        '''
        # initialize log
        self.log = {
            'timestamp': [],
            'obs': [],
            'action': [],
            'reward': [],
        }
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        '''
        @brief:
            get the data points stored in feature buffers for all servers w/ shape [N_AS, SEQ_LEN, N_FEATURES]
        @return:
            a tuple that contains (new observation, list of active AS id)
        '''
        # create a mask that will zero-out all features for inactive AS
        obs = np.zeros((self.shm_m.shm_n_bin, SEQ_LEN, len(FEATURES2USE)))

        # get a list of seq number from one active AS
        active_as, sid = self.shm_m.get_current_active_as()
        # get the mask for the rows to be taken from feature buffers
        feature_sid_mask = tuple(
            [np.arange(sid - SEQ_LEN, sid) % self.shm_m.shm_seq_len])
        for asid in active_as:
            ptr = self.shm_m.ptrs["feature_buffers"][asid]
            feature_buffer = np.array(self.shm_m.unpack_mem(
                ptr)).reshape(-1, len(self.shm_m.feature_keys))
            obs[asid] = feature_buffer[feature_sid_mask][:, self.feature_mask]
        return (obs, active_as)

    def _take_action(self, action):
        '''
        @brief:
            set the weights in LB node according to action_type ('score' - the lower the better, or 'alias' weighted sampling)
        '''
        if self.action_type == 'score':
            self.shm_m.register_as_score(self.current_step, action)
        elif self.action_type == 'alias':
            alias = [(1., 0)] * self.shm_m.shm_n_bin
            non_zero_weights = [v for v in action if v > 0]
            non_zero_weights_id = [i for i, v in enumerate(action) if v > 0]

            non_zero_alias = sm.gen_alias(non_zero_weights)
            # print("======")
            # print(">> bin:     {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d}".format(*non_zero_weights_id))
            # print(">> weights: {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f}".format(*non_zero_weights))
            # print(">> odds:    {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f}".format(*[_[0] for _ in non_zero_alias]))
            # print(">> alias:   {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d}".format(*[_[1] for _ in non_zero_alias]))

            for _alias, asid in zip(non_zero_alias, non_zero_weights_id):
                alias[asid] = (_alias[0], _alias[1])
            self.shm_m.register_as_alias(self.current_step, alias)

    def _calcul_reward(self, action, obs, active_as):
        '''
        @brief:
            calculate reward based on the fairness of recent reservoir observation (flow duration avg)
        '''
        # use negative reservoir_flow_duration_avg_decay times their corresponding AS weights (weighted average) as reward for now
        return -np.mean(obs[active_as][:, -1, self.reward_field_idx] * np.array(action)[active_as])

    def step(self, action):
        '''
        @brief:
            execute one time step within the environment
        @params:
            action: a list of weights 
        '''
        self._take_action(action)

        self.current_step += 1

        # sleep for a while
        time.sleep(self.interval)

        # get next states and reward
        obs = self._next_observation()
        reward = self._calcul_reward(action, *obs)

        self.log['timestamp'].append(time.time())
        self.log['obs'].append(obs)
        self.log['action'].append(action)
        self.log['reward'].append(reward)

        return obs, reward, False, {}  # for now done is always False, and info is empty

    def dev(self):
        '''
        @brief:
            for off-line dev purpose
        '''
        print("=== Yo! This is dev mode! ===")
        print(self.observation_space)
        print(self.observation_space[0].low)
        # print(self.observation_space.high[20:30])
        # print(self._next_observation())

    def dev_online(self):
        '''
        @brief:
            for on-line dev purpose
        '''
        print("=== Yo! This is dev mode (online)! ===")
        # print(self.observation_space)
        # print(self.observation_space.low[20:30])
        # print(self.observation_space.high[20:30])
        print(self._next_observation()[13])

    def render(self, mode='cli'):
        '''
        @brief:
            Render the environment to the screen (print out the final step for actions)
        '''
        def max_scale(v):
            _max = max(v)
            return np.array(v)/max(1e-6, _max), _max

        for s in self.gt_sockets:
            s.sendall(b'42\n')  # send query to AS and wait for response

        obs_lst, active_as = self._next_observation()
        obs_lst = np.array(obs_lst)
        gt_lst = [0] * len(self.gt_sockets)
        print("=" * 89)
        print("{:<30s}".format(">> Step:"), self.current_step)
        print("{:<30s}".format("Latest Reward:"), self.log['reward'][-1])
        action_scaled, action_max = max_scale(
            np.array(self.log['action'][-1])[active_as])
        flow_dur_decay_scaled, flow_dur_decay_max = max_scale(
            obs_lst[active_as][:, -1, [k for k, v in FEATURES2USE.items()].index('res_flow_duration_avg_decay')])
        fct_decay_scaled, fct_decay_max = max_scale(
            obs_lst[active_as][:, -1, [k for k, v in FEATURES2USE.items()].index('res_fct_avg_decay')])
        print("{:<30s}".format("Last max action:"), action_max)
        print("{:<30s}".format("Latest max flow dur. decay:"), flow_dur_decay_max)
        print("{:<30s}".format("Latest max fct. decay:"), fct_decay_max)
        for i, s in enumerate(self.gt_sockets):
            data = s.recv(24)  # send query to AS and wait for response
            gt_lst[-i-1] = list(struct.unpack('dqii', data))[-2]
        gt_scaled, gt_max = max_scale(gt_lst)
        print("{:<30s}".format("Latest max #apache:"), gt_max)
        if mode == 'ipynb':
            fig = plt.figure(figsize=(16, 4))
            ax = fig.add_subplot(111)
            frame = np.vstack(
                (gt_scaled, action_scaled, flow_dur_decay_scaled, fct_decay_scaled))
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
        elif mode == 'cli':
            print("{:<30s}".format("Latest active ASs:"), ' |'.join(
                [" {:>5d}".format(asid) for asid in active_as]))
            print("{:<30s}".format("#apache:"), ' |'.join(
                [" {:>5d}".format(gt) for gt in gt_lst]))
            print("{:<30s}".format("Last action:"), ' |'.join(
                [" {:>0.3f}".format(_) for _ in np.array(self.log['action'][-1])[active_as]]))
            print("{:<30s}".format("Latest avg. flow dur. decay:"), ' |'.join([" {:>0.3f}".format(
                _) for _ in obs_lst[active_as][:, -1, [k for k, v in FEATURES2USE.items()].index('res_flow_duration_avg_decay')]]))
            print("{:<30s}".format("Latest avg. fct. decay:"), ' |'.join([" {:>0.3f}".format(
                _) for _ in obs_lst[active_as][:, -1, [k for k, v in FEATURES2USE.items()].index('res_fct_avg_decay')]]))


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

parser.add_argument('-i', action='store_true',
                    default=0.25,
                    dest='interval',
                    help='Set sleep interval in env.step() for action to take effect')

parser.add_argument('--version', action='version',
                    version='%(prog)s 1.0')

#--- Macros ---#
frame_idx = 0  # number of iterations
batch_size = 64  # number of batches for training
explore_steps = 100  # number of iterations for pure exploration
update_itr = 1  # how many iterations do we update each time
max_steps = 150  # for dev
render_cycle = 2  # every ${render_cycle} steps, print out once current state

if __name__ == '__main__':
    args = parser.parse_args()

    lbenv = LoadBalanceEnv(args.method, args.interval)
    lbenv.reset()
    active_as, _ = lbenv.shm_m.get_current_active_as()

    for step in range(max_steps):
        if frame_idx > explore_steps:
            # take actions from our policy_net (TBD)
            action = np.zeros(lbenv.shm_m.shm_n_bin)
            action[active_as] = np.random.rand(len(active_as))
        else:
            # randomly sample actions
            action = np.zeros(lbenv.shm_m.shm_n_bin)
            action[active_as] = np.random.rand(len(active_as))
        next_state, reward, _, _ = lbenv.step(action)

        state = next_state
        active_as = state[1]
        frame_idx += 1

        # update SAC (TBD)

        # render
        if frame_idx % render_cycle == 0:
            lbenv.render()
