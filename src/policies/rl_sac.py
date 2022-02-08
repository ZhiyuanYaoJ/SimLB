
import random
import time
import numpy as np
from config.user_conf import FEATURE_AS_ALL, FEATURE_LB_ALL, N_FEATURE_AS, N_FEATURE_LB, ACTION_DIM
from common.entities import lbNode
from policies.model.sac_v2 import *

#--- MACROS ---#
SAC_training_confs = {'hidden_dim': 512,
                      'action_range': 1.,
                      'batch_size': 64,
                      'update_itr': 10,
                      'reward_scale': 10.,
                      'save_interval': 100,  # time interval for saving models, in seconds
                      'AUTO_ENTROPY': True,
                      'model_path': 'rl/sac_v2',
                      }

DETERMINISTIC = False


class rlbSAC(lbNode):
    '''
    @brief:
        RL solution for simulated load balancer with SAC algorithm.
    '''

    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        args = kwargs['args']
        self.last_t0 = self.T0  # time of last timestep
        self.alpha = args.heuristic_alpha  # weights updating alpha
        # init rl agent
        self.logger = init_logger("log/rl.log", "rl-logger")
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.last_state = None
        self.last_action = None
        self.sac_trainer = SAC_Trainer(self.replay_buffer, n_feature_as=N_FEATURE_AS, n_feature_lb=N_FEATURE_LB,
                                       hidden_dim=SAC_training_confs['hidden_dim'], action_range=SAC_training_confs['action_range'],
                                       action_dim=ACTION_DIM, logger=self.logger)
        # load model if possible
        model_path = SAC_training_confs['model_path']
        if args.rl_test:
            self.sac_trainer.load_model(model_path)

    def reset(self):
        super(self.__class__, self).reset()
        self.last_state = None
        self.last_action = None

    def choose_as(self):
        as_id = random.choices(
            self.active_as, [self.weights[as_id_] for as_id_ in self.active_as])
        return as_id[0]

    def generate_weight(self, state):
        '''
        @brief:
            core algorithm for updating weights, in to steps:
                1. generate an inferred new state (weights)
                2. data fusion of new and old states
        '''
        t0 = time.time()
        # step 1: prediction
        new_weights = self.sac_trainer.policy_net.get_action(
            state, deterministic=DETERMINISTIC)

        # step 2: apply weights
        super(self.__class__, self).update_weight_buffer(
            self.alpha*(new_weights+self.weights))
        print("delta t generating weight: ", time.time() - t0)
        return time.time() - t0

    def step(self, ts):
        '''
        @brief:
            one step forward for agent, including:
            1. get reward for last action;
            2. feed samples into buffer;
            3. train agent for once or several times (TODO: assume training is costless since in testbed it will be done in an independent thread)
            4. provide next action for env. 
        '''
        t0 = time.time()  # take the first timestamp

        print('-'*10, ts, '-'*10)
        # step 0: get observation
        state = super(
            self.__class__, self).get_observation(ts, update_res=True)
        active_as, feature_lb, feature_as, ground_truth = state

        # step 1: get reward for last action
        reward = self._calcul_reward(feature_as)

        # step 2: feed data into buffer
        if self.last_state:  # ignore the first step
            self.replay_buffer.push(
                self.last_state, self.last_action, reward, state)

        # step 3
        t1 = time.time()  # take the second timestamp
        print('DEBUG: step 0-2 {:.3f}s'.format(t1-t0))
        self.train()

        # step 4
        print('DEBUG: step 3 train {:.3f}s'.format(time.time()-t1))
        t_gen_weight = self.generate_weight(state)

        # step -1
        self.last_state = state
        self.last_action = self.weights_buffer

        step_delay = t1 - t0 + t_gen_weight  # just ignore train time
        print('DEBUG: step_delay train {:.3f}s'.format(step_delay))

        # save model if necessary
        if t0 - self.last_t0 > SAC_training_confs['save_interval']:
            self.sac_trainer.save_model(SAC_training_confs['model_path'])
            self.last_t0 = t0

        return step_delay

    def train(self):
        '''
        @brief:
            update SAC models with samples from buffer
            TODO: whether update for each step or for each episode, need to be determined
        '''
        if len(self.replay_buffer) > SAC_training_confs['batch_size']:
            for i in range(SAC_training_confs['update_itr']):
                _ = self.sac_trainer.update(
                    SAC_training_confs['batch_size'],
                    reward_scale=SAC_training_confs['reward_scale'],
                    auto_entropy=SAC_training_confs['AUTO_ENTROPY'],
                    target_entropy=-1.*self.action_dim,
                )

    def render(self, ts, update_res=False):

        active_as, feature_lb, feature_as, ground_truth = self.get_observation(
            ts, update_res=True)
        self.logger.info('='*10)
        self.logger.info('{:<30s}'.format('LB Node: ')+'{}'.format(self.id))
        self.logger.info('{:<30s}'.format('Time: ') +
                         '{:.6f}s'.format(time.time()-self.T0))
        self.logger.info('{:<30s}'.format('Sim. Time: ')+'{:.6f}s'.format(ts))
        self.logger.info('{:<30s}'.format('Step: ')+'{}'.format(self.n_update))
        self.logger.info('{:<30s}'.format('Latest Reward: ') +
                         '{}'.format(self._calcul_reward(feature_as)))
        self.logger.info('{:<30s}'.format('feature_lb:')+', '.join(['{}: {:.3f}'.format(
            k, v) for k, v in zip(FEATURE_LB_ALL, feature_lb) if 'iat_f_lb' in k]))
        self.logger.info('{:<30s}'.format('Latest active ASs:')+' |'.join(
            [' {:>7d}'.format(asid) for asid in active_as]))
        self.logger.info('{:<30s}'.format('#apache obs.:')+' |'.join(
            [' {:>7.0f}'.format(feature_as[i, FEATURE_AS_ALL.index('n_flow_on')]) for i in active_as]))
        self.logger.info('{:<30s}'.format('rejected requests.:')+' |'.join(
            [' {:>7.0f}'.format(feature_as[i, FEATURE_AS_ALL.index('n_reject')]) for i in active_as]))
        self.logger.info('{:<30s}'.format('fd_avg.:')+' |'.join(
            [' {:>7.3f}'.format(feature_as[i, FEATURE_AS_ALL.index('fd_avg')]) for i in active_as]))
        self.logger.info('{:<30s}'.format('fd_res_avg.:')+' |'.join(
            [' {:>7.3f}'.format(feature_as[i, FEATURE_AS_ALL.index('fd_res_avg')]) for i in active_as]))
        self.logger.info('{:<30s}'.format('fct_avg.:')+' |'.join(
            [' {:>7.3f}'.format(feature_as[i, FEATURE_AS_ALL.index('fct_avg')]) for i in active_as]))
        self.logger.info('{:<30s}'.format('fct_res_avg.:')+' |'.join(
            [' {:>7.3f}'.format(feature_as[i, FEATURE_AS_ALL.index('fct_res_avg')]) for i in active_as]))
        self.logger.info('{:<30s}'.format('Last action:')+' |'.join(
            [' {:> 7.3f}'.format(self.weights[i]) for i in active_as]))
        self.logger.info('{:<30s}'.format('#apache actual:')+' |'.join(
            [' {:> 7.0f}'.format(ground_truth[i]) for i in active_as]))
