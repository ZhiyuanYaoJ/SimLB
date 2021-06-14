import time
import random
import numpy as np
from common.utils import softmax
from common.entities import NodeLB, namedtuple
from config.global_conf import ACTION_DIM, RENDER, LB_PERIOD, HEURISTIC_FEATURE, HEURISTIC_ALPHA, KF_CONF, B_OFFSET
from common.cons_hash import *

Gaussian = namedtuple('Gaussian', ['mean', 'var'])
Gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])

class NodeLBAquarius(NodeLB):
    '''
    @brief:
        Aquarius: replacing weights by inference based on reservoir-sampled flow durations, soft-update using alpha
    '''

    def __init__(self, id, child_ids, bucket_size=65536, weights=None, max_n_child=ACTION_DIM, T0=time.time(), reward_option=2, ecmp=False, child_prefix='as', po2=False, b_offset=B_OFFSET, debug=0):
        super().__init__(id, child_ids, bucket_size, weights,
                         max_n_child, T0, reward_option, ecmp, child_prefix, debug)
        self.po2 = po2  # power-of-2-choices
        self.b_offset = b_offset
        self.feature2use = HEURISTIC_FEATURE
        self.alpha = HEURISTIC_ALPHA

        assert 0 < self.alpha <= 1

    def step(self, ts, nodes=None):
        '''
        @brief:
            core algorithm for updating weights, in two steps:
                1. generate an inferred new state (weights) from observations (reservoir sampled flow duration)
                2. data fusion of new and old states
        '''
        # step 1: prediction
        obs = self.get_observation(ts)
        new_state = softmax(-obs[self.feature2use][self.child_ids])
        new_weights = np.zeros(self.max_n_child)
        new_weights[self.child_ids] = new_state
        if self.debug > 1:
            print(">> ({:.3f}s) in {}: origin weights {} - new weights {}".format(
                ts, self.__class__, self.weights[self.child_ids], new_weights[self.child_ids]))


        # step 2: apply weights
        self.weights = self.alpha*new_weights+(1-self.alpha)*self.weights
        if self.debug > 1:
            print(">> ({:.3f}s) in {}: updated weights {}".format(
                ts, self.__class__, self.weights[self.child_ids]))
        if RENDER:
            self.render(ts, nodes)
        ts += self.get_process_delay()
        self.register_event(ts, 'lb_update_bucket', {'node_id': self.id})
        self.register_event(ts + LB_PERIOD,
                            'lb_step', {'node_id': self.id})

    def choose_child(self, flow):
        # we still need to generate a bucket id to store the flow
        bucket_id, _ = self._ecmp(
            *flow.fields, self._bucket_table, self._bucket_mask)
        n_flow_on = self._counters['n_flow_on']
        if self.debug > 1:
            print("@nodeHLB {} - n_flow_on: {}".format(self.id, n_flow_on))
        # assert len(set(self.child_ids)) == len(self.child_ids)
        if self.po2:
            n_flow_on_2 = {i: (self.b_offset+n_flow_on[i])/self.weights[i]
                           for i in random.sample(self.child_ids, 2)}
            child_id = min(n_flow_on_2, key=n_flow_on_2.get)
            if self.debug > 1:
                print("n_flow_on chosen {} out of -".format(child_id), n_flow_on_2)
        else:
            n_flow_on = [(self.b_offset+n_flow_on[i])/self.weights[i]
                         for i in self.child_ids]
            min_n_flow = min(n_flow_on)
            n_flow_map = zip(self.child_ids, n_flow_on)
            min_ids = [k for k, v in n_flow_map if v == min_n_flow]
            child_id = random.choice(min_ids)
            if self.debug > 1:
                n_flow_map = zip(self.child_ids, n_flow_on)
                print("n_flow_on chosen minimum {} from {}".format(
                    child_id, '|'.join(['{}: {}'.format(k, v) for k, v in n_flow_map])))
            del n_flow_map
        return child_id, bucket_id

class NodeHLB(NodeLB):
    '''
    @brief:
        Replacing soft-update alpha in Aquarius by Kalman filter
    '''

    def __init__(
            self,
            id,
            child_ids,
            bucket_size=65536,
            weights=None,
            max_n_child=ACTION_DIM,
            T0=time.time(),
            reward_option=2,
            ecmp=False,
            child_prefix='as',
            system_mean=KF_CONF['system_mean'],
            system_std=KF_CONF['system_std'],
            sensor_std=KF_CONF['sensor_std'],
            po2=False,
            b_offset=B_OFFSET,
            debug=0):
        super().__init__(id, child_ids, bucket_size, weights,
                        max_n_child, T0, reward_option, ecmp, child_prefix, debug)

        self.feature2use = HEURISTIC_FEATURE
        self.system_var = system_std**2

        self.process_model = Gaussian(system_mean, self.system_var)
        self.sensor_var = sensor_std**2
        self.reset_local()
        
        self.po2 = po2  # power-of-2-choices
        self.b_offset = b_offset

    def predict(self, pos, movement):
        return Gaussian(pos.mean + movement.mean, pos.var + movement.var)

    def update(self, prior, likelihood):
        mean = (prior.var * likelihood.mean + likelihood.var *
                prior.mean) / (prior.var + likelihood.var)
        variance = (prior.var * likelihood.var) / (prior.var + likelihood.var)
        # print(">> Kalman Gain = {}".format(variance/(variance+likelihood.var)))
        return Gaussian(mean, variance)

    def reset_local(self):
        self.xs = []
        for i in range(self.max_n_child):
            if i in self.child_ids:
                # initialize state estimation
                self.xs.append(
                    Gaussian(KF_CONF['init_mean'], KF_CONF['init_std']**2))
            else:
                self.xs.append(Gaussian(0., 0.,))
        self.ps = np.zeros(self.max_n_child)  # variance of estimation
        self.zs = np.zeros(self.max_n_child)  # latest measurements

    def reset(self):
        super().reset()
        self.reset_local()

    def step(self, ts, nodes=None):
        '''
        @brief:
            core algorithm for updating weights, in two steps:
                1. generate an inferred new state (weights) from observations (reservoir sampled flow duration)
                2. data fusion of new and old states
        '''
        # step 1: prediction
        obs = self.get_observation(ts)
        feature = obs[self.feature2use][self.child_ids]
        # print(">> {} : ".format(self.feature2use)+str(feature))

        new_state = feature/(feature.mean()+1e-9)
        self.zs[self.child_ids] = new_state  # update measurement array

        new_weights = np.zeros(self.max_n_child)
        for child_id in self.child_ids:
            prior = self.predict(self.xs[child_id], self.process_model)
            self.xs[child_id] = self.update(
                prior, Gaussian(self.zs[child_id], self.sensor_var))
            new_weights[child_id] = self.xs[child_id].mean
            self.ps[child_id] = self.xs[child_id].var
        if self.debug > 1:
            print(">> ({:.3f}s) in {}: origin weights {} - new weights {}".format(
                ts, self.__class__, self.weights[self.child_ids], new_weights[self.child_ids]))

        print(">> Kalman Gain = {}".format(
            self.xs[child_id].var/(self.xs[child_id].var+self.sensor_var)))

        # step 2: apply weights
        self.weights[self.child_ids] = softmax(-new_weights[self.child_ids])

        # print(">> ({:.3f}s) in {}: new weights {}".format(
        #     ts, self.__class__, self.weights[self.child_ids]))
        
        if RENDER:
            self.render(ts, nodes)
        ts += self.get_process_delay()
        self.register_event(ts, 'lb_update_bucket', {'node_id': self.id})
        self.register_event(ts + LB_PERIOD,
                            'lb_step', {'node_id': self.id})

        print(">> ({:.3f}s) in {}: new weights {}".format(
                ts, self.__class__, new_weights[self.child_ids]))

    def add_child(self, child_id, weights=None):
        super().add_child(child_id, weights)
        for i in child_id:
            self.xs[i] = Gaussian(KF_CONF['init_mean'], KF_CONF['init_std']**2)
        self.ps[child_id] = 0  # variance of estimation
        self.zs[child_id] = 0  # latest measurements

    def choose_child(self, flow, nodes=None, ts=None):
        # we still need to generate a bucket id to store the flow
        bucket_id, _ = self._ecmp(
            *flow.fields, self._bucket_table, self._bucket_mask)
        n_flow_on = self._counters['n_flow_on']
        if self.debug > 1:
            print("@nodeHLB {} - n_flow_on: {}".format(self.id, n_flow_on))
        # assert len(set(self.child_ids)) == len(self.child_ids)

        if self.po2:
            n_flow_on_2 = {i: (self.b_offset+n_flow_on[i])/self.weights[i]
                           for i in random.sample(self.child_ids, 2)}
            child_id = min(n_flow_on_2, key=n_flow_on_2.get)
            if self.debug > 1:
                print("n_flow_on chosen {} out of -".format(child_id), n_flow_on_2)
        else:
            score = [(self.b_offset+n_flow_on[i])/self.weights[i]
                            for i in self.child_ids]
            
            score = [(self.b_offset+score[i])/self.weights[i]
                         for i in self.child_ids]
            min_n_flow = min(score)
            n_flow_map = zip(self.child_ids, score)
            min_ids = [k for k, v in n_flow_map if v == min_n_flow]
            child_id = random.choice(min_ids)
            if self.debug > 1:
                n_flow_map = zip(self.child_ids, score)
                print("n_flow_on chosen minimum {} from {}".format(
                    child_id, '|'.join(['{}: {}'.format(k, v) for k, v in n_flow_map])))
            del n_flow_map
        return child_id, bucket_id


class NodeHLBada(NodeHLB):
    '''
    @brief:
        Replacing soft-update alpha in Aquarius by Kalman filter with an adaptive measuremnet error
    '''

    def __init__(
            self,
            id,
            child_ids,
            bucket_size=65536,
            weights=None,
            max_n_child=ACTION_DIM,
            T0=time.time(),
            reward_option=2,
            ecmp=False,
            child_prefix='as',
            lb_period=LB_PERIOD,
            system_mean=KF_CONF['system_mean'],
            system_std=KF_CONF['system_std'],
            sensor_std=KF_CONF['sensor_std'],
            po2=False,
            b_offset=B_OFFSET,
            debug=0):
        super().__init__(id, child_ids, bucket_size, weights,
                         max_n_child, T0, reward_option, ecmp, child_prefix, system_mean, system_std, sensor_std, debug, lb_period)
        self.po2 = po2  # power-of-2-choices
        self.b_offset = b_offset

    
    def step(self, ts, nodes=None):
        '''
        @brief:
            core algorithm for updating weights, in three steps:
                1. generate an inferred new state (weights) from observations (reservoir sampled flow duration)
                2. data fusion of new and old states
                3. update sensor error based on reservoir sample variance
        '''
        # step 1: prediction
        obs = self.get_observation(ts)
        feature = obs[self.feature2use][self.child_ids]

        print(">> {} : ".format(self.feature2use)+str(feature))

        new_state = feature/(feature.mean()+1e-9)
        self.zs[self.child_ids] = new_state  # update measurement array

        new_weights = np.zeros(self.max_n_child)
        for child_id in self.child_ids:
            prior = self.predict(self.xs[child_id], self.process_model)
            self.xs[child_id] = self.update(
                prior, Gaussian(self.zs[child_id], self.sensor_var))
            new_weights[child_id] = self.xs[child_id].mean
            self.ps[child_id] = self.xs[child_id].var
        if self.debug > 1:
            print(">> ({:.3f}s) in {}: origin weights {} - new weights {}".format(
                ts, self.__class__, self.weights[self.child_ids], new_weights[self.child_ids]))

        # print(">> Kalman Gain = {}".format(
        #     self.xs[child_id].var/(self.xs[child_id].var+self.sensor_var)))

        # step 2: apply weights
        self.weights[self.child_ids] = softmax(-new_weights[self.child_ids])

        print(">> ({:.3f}s) in {}: new weights".format(
            ts, self.__class__)+str(self.weights[self.child_ids]))

        if RENDER:
            self.render(ts, nodes)
        ts += self.get_process_delay()
        self.register_event(ts, 'lb_update_bucket', {'node_id': self.id})
        self.register_event(ts + LB_PERIOD,
                            'lb_step', {'node_id': self.id})

        # step 3: update sensor variance
        self.sensor_var = 0.99*self.sensor_var + 0.01 * \
            new_state.var() + self.system_var

class NodeLBHermes(NodeLB):
    '''
    @brief:
        Hermes: distributes load depending on choices made by servers
    '''

    def __init__(self, id, child_ids, bucket_size=65536, weights=None, max_n_child=ACTION_DIM, T0=time.time(), reward_option=2, ecmp=False, child_prefix='as', po2=False, debug=0):
        super().__init__(id, child_ids, bucket_size, weights,
                         max_n_child, T0, reward_option, ecmp, child_prefix, debug)

        self.perm = [[9, 13], [14, 19], [3, 2], [1, 14], [16, 17], [18, 12], [1, 11], [7, 15], [0, 10], [6, 11], [4, 8], [2, 4], [9, 12], [8, 7], [15, 19], [5, 16], [6, 18], [13, 0], [3, 10], [5, 17]]
        self.M = len(self.child_ids) #size lookup table / for now similar to child length
        self.N = len(self.child_ids) # number of AS
        self.cons_hash = Cons_hash(_M=self.M,_N=self.N,_perm=self.perm)
        self.lookup_table = self.cons_hash.compute_table()
        
        self.choice = [0 for i in range(self.M)]
        self.scores = [0 for i in range(self.N)] 

    def choose_child(self, flow, nodes, ts):
        # we still need to generate a bucket id to store the flow
        bucket_id, _ = self._ecmp(
            *flow.fields, self._bucket_table, self._bucket_mask)
        n_flow_on = self._counters['n_flow_on']

        if self.debug > 1:
            print("=== proc_time (@{}) ===".format(self.id))
            print("score:", [nodes['{}{:d}'.format(self.child_prefix,i)].get_avg_proc_time(ts)
                         for i in self.child_ids])
        gt = self.get_ground_truth(nodes, ts, flow)
        
        if self.debug > 1:
            print("n_flow:", gt['n_flow'])
            print("t_remain:", gt['t_remain'])

            print("@nodeLBHermes {} - n_flow_on: {}".format(self.id, n_flow_on))
        # assert len(set(self.child_ids)) == len(self.child_ids)
        ind = random.randint(0,self.M-1)
        child_id = self.lookup_table[ind][self.choice[ind]]
        score_other = self.scores[self.lookup_table[ind][self.choice[ind]^1]]
        _node = nodes['{}{:d}'.format(self.child_prefix,child_id)]
        score,choice = _node.update_score(score_other,self.choice[ind],ts)
        self.scores[child_id] = score
        self.choice[ind] = choice 

        debug_carmine = False
        
        if debug_carmine:
            print("**** RANDOM INDEX {}".format(ind))
            print("++++ CHILD ID {}".format(child_id))
            print("---- TABLE ---- ",self.choice)
            print("///// ",self.lookup_table)
            print(">>>> ",[ nodes['{}{:d}'.format(self.child_prefix,i)].get_n_flow_on()  for i in self.child_ids])
            print("<<<< ",self.scores)
        if self.debug > 1:
            print("n_flow_on chosen {} out of -".format(child_id), n_flow_on_2)
        
        return child_id, bucket_id
