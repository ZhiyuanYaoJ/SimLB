
import random
import time
import numpy as np
from config.global_conf import ACTION_DIM, RENDER, DISPLAY, FEATURE_AS_ALL, FEATURE_LB_ALL, N_FEATURE_AS, N_FEATURE_LB, B_OFFSET, HEURISTIC_ALPHA, LB_PERIOD, HIDDEN_DIM, REWARD_OPTION, REWARD_FEATURE,DEBUG
from common.entities import NodeLB
from policies.model.sac_v2 import *


from functools import wraps
i=0
t0=0
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        #print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        global t0, i
        if total_time>0.0:
            t0 += total_time
            i +=1
            print('Result {}'.format(total_time))
        return result
    return timeit_wrapper

#--- MACROS ---#
SAC_training_confs = {'hidden_dim': HIDDEN_DIM,
                      'action_range': 1.,
                      'batch_size': 64,
                      'update_itr': 10,
                      'reward_scale': 10.,
                      'save_interval': 30,  # time interval for saving models, in seconds
                      'AUTO_ENTROPY': True,
                      'model_path': 'models2/sac_v2',
                      }

DETERMINISTIC = False

class NodeRLBSAC_Tiny(NodeLB):
    '''
    @brief:
        RL solution for simulated load balancer with SAC model.
    '''

    def __init__(
            self, 
            id,
            child_ids, # ids of the children server nodes connected to the LB
            bucket_size=65536, # bucket table size for WCMP
            weights=None, # initial weights
            max_n_child=ACTION_DIM, # action space size [ACTION_DIM, ]
            T0=time.time(),
            reward_option=REWARD_OPTION,
            ecmp=False,
            child_prefix='as',
            b_offset=B_OFFSET,
            logger_dir="log/rl.log",
            rl_test=False,
            layer=1,
            lb_period=LB_PERIOD,
            SAC_training_confs_=SAC_training_confs,
            debug=0
            ):
        super().__init__(id, child_ids, bucket_size, weights, 
                         max_n_child, T0, reward_option, ecmp, child_prefix, layer, lb_period = lb_period, debug = debug)
        self.last_save_t = self.T0  # time of last timestep
        self.alpha = HEURISTIC_ALPHA  # weights updating alpha
        self.rl_test = rl_test
        self.b_offset = b_offset
        # init rl agent
        self.logger = init_logger(logger_dir, "rl-logger")
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE * len(self.child_ids))
        self.last_state = None
        self.last_action = None
        self.action_dim = 1
        global SAC_training_confs
        SAC_training_confs = SAC_training_confs_
        self.sac_trainer = SAC_Trainer(self.replay_buffer, n_feature_as=N_FEATURE_AS, n_feature_lb=N_FEATURE_LB + max_n_child,
                                       hidden_dim=SAC_training_confs['hidden_dim'], action_range=SAC_training_confs['action_range'],
                                       action_dim=self.action_dim, logger=self.logger)
        
    def reset(self):
        super().reset()
        self.last_state = None
        self.last_action = None
        # load model if possible
        if self.rl_test:
            model_path = SAC_training_confs['model_path']
            self.sac_trainer.load_model(model_path)

    def choose_child(self, flow, nodes=None, ts=None):
        # we still need to generate a bucket id to store the flow
        bucket_id, _ = self._ecmp(
            *flow.fields, self._bucket_table, self._bucket_mask)
        n_flow_on = self._counters['n_flow_on']
        if self.debug > 1:
            print("@nodeLBSAC {} - n_flow_on: {}".format(self.id, n_flow_on))
        # assert len(set(self.child_ids)) == len(self.child_ids)

        score = [(self.b_offset+n_flow_on[i])/self.weights[i] for i in self.child_ids]

#       score = [(self.b_offset+score[i])/self.weights[i] for i in self.child_ids]

        
        min_n_flow = min(score)
        n_flow_map = zip(self.child_ids, score)
        min_ids = [k for k, v in n_flow_map if v == min_n_flow]
        child_id = random.choice(min_ids)
        if self.debug > 1:
            n_flow_map = zip(self.child_ids, score)
            print("n_flow_on chosen minimum {} from {}".format(
                child_id, '|'.join(['{}: {}'.format(k, v) for k, v in n_flow_map])))
            
            
        return child_id, bucket_id


    def calcul_reward(self, ts, reward_field=REWARD_FEATURE):
        '''
        @brief: calculate reward using a given REWARD FEATURE
        '''
        feature_all = self.get_observation(ts)
        feature_reward = feature_all[reward_field][self.child_ids]
        size = len(feature_reward)
        feature_reward = np.array(feature_reward)
        A = np.ones(size)-np.eye(size)
        B = 1/(size-1) * feature_reward @ A
        reward = {}
        for i,a in enumerate(self.child_ids):
            reward[a] = self.reward_fn([feature_reward[i], B[i]])

        return reward
    
    def get_state(self, ts, nodes=None, child_id = None):
        '''
        @brief:
            get state that matches the SAC model format
        '''
        obs = self.get_observation(ts)
        all_feature_as = np.array([obs[k] for k in FEATURE_AS_ALL]).T
        one_hot = [0]*self.max_n_child
        one_hot[child_id] = 1
        feature_as = np.array([[obs[k][child_id] for k in FEATURE_AS_ALL]])
        #feature_lb = [obs[k] for k in FEATURE_LB_ALL] + list(feature_as[self.child_ids].mean(axis=0)) # feature_lb has lb feature + averaged as feature
        feature_lb = [obs[k] for k in FEATURE_LB_ALL] + one_hot + list(all_feature_as[self.child_ids].mean(axis=0)) # feature_lb has lb feature + averaged as feature
        if False:
            t_rest_all = np.zeros(self.max_n_child)
            t_rest_all[self.child_ids] = [nodes['{}{:d}'.format(self.child_prefix, i)].get_t_rest_total(ts) for i in self.child_ids]
        t_rest_all = None    
        
        return ([0], feature_lb, feature_as, t_rest_all) # gt set to rest time

    def generate_weight(self, state, child_id = None):
        '''
        @brief:
            core algorithm for updating weights, in to steps:
                1. generate an inferred new state (weights)
                2. data fusion of new and old states
        '''
        t0 = time.time()
        
        new_weights = self.sac_trainer.policy_net.get_action(state, deterministic=DETERMINISTIC)
        # softly update weights
        self.weights[child_id] = self.alpha*new_weights+(1-self.alpha)*self.weights[child_id]
        return time.time() - t0

    def step(self, ts, nodes=None):
        '''
        @brief:
            one step forward for agent, including:
            1. get reward for last action;
            2. feed samples into buffer;
            3. train agent for once or several times (TODO: assume training is costless since in testbed it will be done in an independent thread)
            4. provide next action for env.
        '''
        t0 = time.time()  # take the first timestamp

        # step 0: get state
        state = {}
        for i in self.child_ids:
            state[i] = self.get_state(ts, nodes=nodes, child_id=i)

        # step 1: get reward for last action
        reward = self.calcul_reward(ts)

        # step 2: feed data into buffer
        if self.last_state:  # ignore the first step
            for i in self.child_ids:
                self.replay_buffer.push(
                    self.last_state[i], [self.last_action[i]], reward[i], state[i])

        # step 3
        t1 = time.time()  # take the second timestamp
        self.train()

        # step 4
        t_gen_weight = 0
        for i in self.child_ids:
            t_gen_weight += self.generate_weight(state[i], child_id=i)

        # step -1
        self.last_state = state
        self.last_action = self.weights

        step_delay = t1 - t0 + t_gen_weight  # just ignore train time
        
        if self.debug > 1:
            print('DEBUG: step_delay train {:.3f}s'.format(step_delay))

        # save model if necessary
        if t0 - self.last_save_t > SAC_training_confs['save_interval']:
            #self.sac_trainer.save_model(SAC_training_confs['model_path'])
            self.last_save_t = t0

        ts += step_delay
        self.register_event(ts, 'lb_update_bucket', {'node_id': self.id})
        self.register_event(ts + self.lb_period,
                            'lb_step', {'node_id': self.id})
        if RENDER:
            self.render(ts, state)

        if DISPLAY>0 and self.layer==1 and self.debug >1:
            # print(">> ({:.3f}s) in {}: new weights {}".format(
                # ts, self.__class__, self.weights[self.child_ids]))
            print("new weights {}".format(self.weights[self.child_ids]))
            print(' ')

    def train(self):
        '''
        @brief:
            update SAC models with samples from buffer
            TODO: whether update for each step or for each episode, need to be determined
        '''
        if len(self.replay_buffer) <2 : return
        if len(self.replay_buffer) < SAC_training_confs['batch_size'] * len(self.child_ids):
            batch_size = len(self.replay_buffer)
        else: batch_size = SAC_training_confs['batch_size'] * len(self.child_ids)
        for i in range(SAC_training_confs['update_itr']):
            _ = self.sac_trainer.update(
                batch_size,
                reward_scale=SAC_training_confs['reward_scale'],
                auto_entropy=SAC_training_confs['AUTO_ENTROPY'],
                target_entropy=-1.*len(self.child_ids),
            )

    def render(self, ts, state):

        active_as, feature_lb, feature_as, ground_truth = state
        self.logger.info('='*10)
        self.logger.info('{:<30s}'.format('LB Node: ')+'{}'.format(self.id))
        self.logger.info('{:<30s}'.format('Time: ') +
                         '{:.6f}s'.format(time.time()-self.T0))
        self.logger.info('{:<30s}'.format('Sim. Time: ')+'{:.6f}s'.format(ts))
        self.logger.info('{:<30s}'.format('Latest Reward: ') +
                         '{}'.format(self.calcul_reward(ts)))
        self.logger.info('{:<30s}'.format('feature_lb:')+', '.join(['{}: {:.3f}'.format(
            k, v) for k, v in zip(FEATURE_LB_ALL, feature_lb) if 'iat_f_lb' in k]))
        self.logger.info('{:<30s}'.format('Latest active ASs:')+' |'.join(
            [' {:>7d}'.format(asid) for asid in active_as]))
        self.logger.info('{:<30s}'.format('#apache obs.:')+' |'.join(
            [' {:>7.0f}'.format(feature_as[i, FEATURE_AS_ALL.index('n_flow_on')]) for i in active_as]))
        self.logger.info('{:<30s}'.format('fd_res_avg.:')+' |'.join(
            [' {:>7.3f}'.format(feature_as[i, FEATURE_AS_ALL.index('res_fd_avg')]) for i in active_as]))
        self.logger.info('{:<30s}'.format('fct_res_avg.:')+' |'.join(
            [' {:>7.3f}'.format(feature_as[i, FEATURE_AS_ALL.index('res_fct_avg')]) for i in active_as]))
        self.logger.info('{:<30s}'.format('Last action:')+' |'.join(
            [' {:> 7.3f}'.format(self.weights[i]) for i in active_as]))
        self.logger.info('{:<30s}'.format('remaining time to process:')+' |'.join(
            [' {:> 7.0f}'.format(ground_truth[i]) for i in active_as]))
