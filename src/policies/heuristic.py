import time
import random
import numpy as np
from common.utils import hash_2tuple, softmax
from common.entities import NodeLB, SamplingBuffer, namedtuple
from config.global_conf import ACTION_DIM, RENDER, LB_PERIOD, HEURISTIC_FEATURE, HEURISTIC_ALPHA, KF_CONF, B_OFFSET
from common.cons_hash import *
from common.alias_method import *
from collections import Counter

Gaussian = namedtuple('Gaussian', ['mean', 'var'])
Gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])

class NodeLBAquarius(NodeLB):
    '''
    @brief:
        Aquarius: replacing weights by inference based on reservoir-sampled flow durations, soft-update using alpha
    '''

    def __init__(self, id, child_ids, bucket_size=65536, weights=None, max_n_child=ACTION_DIM, T0=time.time(), reward_option=2, ecmp=False, child_prefix='as', po2=False, b_offset=B_OFFSET, debug=0):
        super().__init__(id, child_ids, bucket_size, weights,
                         max_n_child, T0, reward_option, ecmp, child_prefix, debug = debug)
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
                        max_n_child, T0, reward_option, ecmp, child_prefix, debug = debug)

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
                         max_n_child, T0, reward_option, ecmp, child_prefix, debug = debug)

        self.perm = [[22, 26, 50, 23, 29, 40, 57, 43, 21, 12, 20, 19, 39, 24, 44, 4, 54, 1, 61, 49, 45, 38, 7, 46, 2, 17, 42, 13, 58, 33, 32, 47, 53, 31, 11, 8, 37, 3, 18, 9, 36, 62, 35, 27, 34, 6, 30, 51, 60, 63, 28, 52, 14, 56, 15, 10, 48, 5, 0, 41, 59, 25, 16, 55], [49, 23, 46, 40, 54, 12, 0, 35, 37, 17, 60, 4, 58, 2, 34, 53, 9, 51, 61, 57, 36, 14, 63, 30, 22, 10, 19, 29, 18, 38, 7, 11, 47, 59, 45, 21, 3, 50, 15, 39, 48, 20, 8, 1, 56, 24, 16, 5, 41, 31, 44, 43, 33, 52, 25, 42, 32, 62, 26, 28, 27, 13, 55, 6], [4, 29, 40, 12, 27, 33, 59, 56, 50, 2, 24, 39, 62, 16, 22, 36, 41, 52, 47, 9, 60, 54, 1, 44, 34, 35, 46, 58, 10, 48, 31, 21, 11, 19, 7, 37, 18, 13, 61, 26, 5, 14, 8, 25, 49, 55, 63, 45, 23, 53, 3, 32, 0, 17, 28, 42, 15, 51, 43, 57, 20, 38, 6, 30], [53, 9, 55, 14, 41, 29, 25, 62, 31, 50, 10, 8, 32, 63, 48, 37, 12, 60, 30, 19, 28, 34, 26, 24, 0, 38, 58, 51, 35, 3, 7, 57, 1, 49, 4, 59, 20, 21, 43, 6, 36, 42, 16, 15, 54, 11, 39, 47, 17, 27, 46, 52, 18, 22, 45, 40, 61, 23, 56, 13, 5, 2, 33, 44], [29, 23, 9, 38, 26, 62, 15, 60, 32, 45, 17, 44, 8, 43, 53, 10, 11, 61, 6, 37, 59, 22, 7, 5, 27, 57, 2, 35, 18, 20, 25, 39, 12, 13, 55, 30, 40, 4, 21, 34, 16, 47, 51, 42, 49, 52, 58, 36, 3, 33, 56, 48, 28, 19, 41, 63, 46, 54, 31, 24, 1, 0, 14, 50], [62, 31, 6, 20, 61, 50, 27, 9, 2, 52, 5, 45, 60, 28, 38, 0, 30, 23, 21, 57, 24, 10, 12, 58, 18, 29, 34, 11, 59, 8, 48, 37, 33, 39, 26, 54, 51, 16, 49, 32, 35, 46, 40, 41, 3, 53, 44, 17, 15, 55, 22, 25, 4, 19, 7, 43, 63, 36, 1, 14, 13, 47, 56, 42], [56, 17, 49, 54, 38, 27, 10, 3, 47, 23, 43, 26, 7, 28, 42, 1, 13, 18, 40, 16, 32, 15, 9, 36, 59, 12, 52, 21, 37, 39, 53, 20, 60, 30, 61, 58, 55, 6, 22, 34, 2, 8, 0, 5, 14, 11, 44, 24, 63, 29, 41, 46, 33, 31, 35, 57, 62, 51, 19, 25, 48, 45, 4, 50], [37, 51, 62, 61, 58, 12, 6, 20, 43, 23, 8, 34, 59, 31, 52, 38, 60, 15, 19, 41, 35, 40, 32, 48, 13, 14, 7, 53, 42, 45, 11, 54, 24, 0, 28, 39, 44, 16, 25, 9, 18, 22, 29, 26, 63, 5, 36, 2, 50, 57, 3, 55, 46, 33, 17, 49, 47, 21, 27, 56, 30, 1, 10, 4], [40, 6, 28, 17, 41, 9, 3, 57, 22, 44, 15, 38, 62, 27, 4, 59, 13, 1, 55, 56, 20, 11, 31, 36, 53, 16, 29, 33, 34, 48, 46, 23, 5, 14, 12, 63, 32, 39, 0, 51, 47, 26, 37, 42, 45, 24, 35, 10, 2, 49, 30, 21, 8, 7, 50, 18, 60, 19, 54, 43, 58, 52, 61, 25], [36, 5, 29, 53, 37, 50, 34, 39, 12, 43, 0, 21, 16, 46, 13, 28, 9, 48, 19, 35, 8, 23, 14, 44, 17, 57, 20, 47, 18, 15, 10, 4, 31, 56, 52, 55, 59, 51, 27, 58, 41, 62, 60, 25, 3, 45, 26, 2, 7, 24, 33, 40, 42, 11, 30, 63, 38, 61, 32, 1, 54, 6, 49, 22], [4, 61, 36, 43, 15, 27, 9, 3, 5, 34, 51, 40, 22, 13, 44, 23, 30, 10, 58, 54, 53, 1, 31, 35, 60, 8, 37, 20, 49, 11, 47, 18, 45, 17, 24, 6, 62, 21, 56, 38, 46, 28, 26, 63, 48, 12, 52, 25, 19, 33, 50, 57, 7, 14, 41, 16, 29, 0, 55, 39, 2, 59, 42, 32], [20, 13, 1, 21, 19, 56, 45, 11, 38, 26, 51, 12, 8, 18, 24, 36, 34, 47, 32, 6, 30, 31, 62, 4, 52, 2, 23, 49, 7, 54, 63, 10, 9, 25, 33, 29, 55, 27, 46, 35, 14, 40, 0, 53, 39, 44, 28, 60, 61, 22, 50, 59, 58, 3, 5, 15, 37, 43, 16, 41, 42, 48, 17, 57], [61, 3, 33, 46, 58, 19, 42, 26, 53, 62, 35, 50, 32, 7, 54, 29, 5, 37, 43, 2, 0, 47, 17, 51, 45, 22, 15, 4, 52, 10, 38, 1, 12, 24, 18, 39, 23, 55, 9, 60, 57, 14, 36, 6, 34, 28, 11, 63, 8, 13, 20, 48, 31, 25, 49, 44, 41, 30, 59, 27, 56, 21, 40, 16], [45, 44, 43, 29, 14, 53, 17, 52, 22, 5, 19, 35, 2, 62, 46, 10, 54, 8, 30, 34, 1, 57, 47, 6, 37, 13, 9, 50, 42, 49, 25, 27, 59, 38, 56, 3, 51, 15, 12, 40, 60, 0, 18, 63, 26, 48, 36, 20, 23, 61, 11, 4, 7, 55, 21, 31, 28, 39, 58, 33, 16, 32, 24, 41], [35, 4, 58, 6, 56, 27, 18, 12, 45, 25, 50, 17, 38, 24, 59, 33, 43, 46, 11, 52, 0, 20, 3, 15, 55, 37, 1, 42, 9, 26, 32, 51, 28, 31, 61, 47, 30, 5, 60, 54, 19, 49, 21, 16, 39, 23, 44, 22, 36, 10, 14, 29, 62, 8, 34, 41, 7, 63, 13, 48, 57, 2, 40, 53], [51, 21, 35, 49, 45, 56, 3, 9, 43, 50, 46, 32, 23, 62, 18, 44, 53, 48, 27, 19, 26, 38, 52, 16, 4, 58, 10, 5, 40, 59, 1, 28, 47, 61, 54, 22, 29, 37, 24, 12, 33, 0, 55, 34, 60, 31, 13, 2, 42, 63, 17, 20, 15, 57, 41, 6, 30, 36, 39, 14, 11, 7, 8, 25], [55, 49, 30, 44, 54, 2, 37, 4, 0, 10, 6, 16, 45, 60, 22, 36, 50, 29, 41, 63, 13, 25, 5, 19, 46, 57, 3, 56, 9, 40, 52, 61, 20, 14, 7, 48, 31, 58, 51, 59, 17, 1, 21, 53, 26, 24, 43, 12, 47, 38, 35, 27, 39, 15, 11, 33, 34, 18, 32, 62, 42, 23, 28, 8], [46, 1, 8, 27, 56, 2, 0, 10, 57, 35, 49, 41, 20, 53, 4, 50, 14, 51, 18, 61, 15, 24, 16, 31, 6, 33, 23, 40, 43, 62, 28, 39, 21, 7, 47, 44, 25, 13, 29, 36, 11, 58, 9, 63, 60, 55, 3, 38, 37, 42, 22, 45, 54, 12, 30, 32, 48, 5, 17, 59, 26, 52, 34, 19], [60, 12, 52, 10, 19, 25, 4, 47, 7, 39, 6, 20, 22, 17, 44, 21, 30, 49, 35, 3, 54, 31, 33, 45, 11, 18, 51, 36, 57, 15, 50, 40, 37, 48, 46, 0, 53, 8, 16, 27, 9, 2, 63, 26, 38, 28, 62, 14, 23, 13, 34, 61, 59, 43, 1, 5, 29, 32, 41, 58, 56, 42, 24, 55], [44, 29, 59, 5, 24, 57, 62, 20, 41, 53, 12, 14, 56, 6, 13, 22, 4, 25, 39, 10, 21, 37, 0, 16, 61, 63, 48, 27, 11, 49, 23, 34, 58, 18, 45, 8, 33, 31, 55, 2, 60, 1, 51, 3, 52, 54, 46, 19, 43, 30, 38, 17, 50, 28, 15, 7, 26, 35, 47, 36, 32, 42, 9, 40], [10, 28, 25, 24, 36, 14, 41, 13, 7, 29, 0, 2, 56, 50, 5, 30, 61, 22, 57, 20, 53, 47, 1, 32, 51, 23, 39, 37, 17, 35, 15, 9, 33, 11, 4, 40, 8, 58, 52, 55, 26, 27, 12, 31, 18, 43, 46, 49, 59, 34, 38, 63, 60, 42, 19, 45, 3, 21, 6, 16, 62, 54, 48, 44], [3, 36, 53, 46, 27, 22, 4, 24, 39, 45, 57, 32, 44, 41, 33, 14, 52, 19, 38, 55, 60, 31, 50, 40, 59, 48, 17, 2, 6, 21, 34, 30, 62, 1, 25, 54, 15, 37, 56, 51, 42, 11, 29, 28, 23, 13, 26, 20, 35, 58, 8, 10, 49, 9, 5, 18, 16, 63, 7, 12, 61, 0, 43, 47], [34, 15, 20, 39, 50, 22, 47, 63, 18, 0, 31, 2, 30, 11, 43, 19, 60, 7, 61, 5, 24, 26, 40, 45, 4, 16, 27, 32, 54, 33, 46, 48, 3, 36, 14, 9, 38, 25, 62, 10, 8, 17, 44, 51, 42, 6, 41, 53, 28, 49, 58, 59, 29, 13, 1, 21, 56, 37, 35, 57, 52, 55, 23, 12], [30, 25, 34, 36, 24, 7, 40, 10, 32, 37, 27, 61, 51, 31, 54, 46, 38, 4, 8, 41, 42, 6, 33, 12, 49, 55, 50, 48, 56, 39, 1, 59, 0, 21, 17, 13, 43, 11, 44, 9, 23, 16, 53, 15, 28, 47, 63, 3, 58, 20, 5, 14, 62, 29, 45, 57, 22, 52, 19, 18, 26, 2, 60, 35], [0, 49, 40, 51, 53, 50, 5, 22, 3, 48, 57, 26, 56, 46, 33, 44, 14, 9, 54, 10, 20, 11, 45, 62, 63, 8, 19, 2, 30, 16, 36, 31, 23, 29, 47, 13, 17, 55, 4, 59, 32, 6, 61, 43, 38, 58, 39, 41, 24, 12, 28, 7, 42, 60, 15, 34, 37, 27, 25, 35, 21, 1, 52, 18], [46, 59, 14, 11, 33, 41, 62, 52, 29, 24, 47, 30, 18, 51, 36, 55, 9, 3, 19, 15, 5, 27, 56, 26, 43, 21, 40, 48, 37, 35, 45, 4, 20, 32, 0, 16, 63, 49, 8, 38, 34, 54, 25, 50, 2, 44, 61, 60, 23, 31, 39, 7, 22, 42, 13, 28, 12, 17, 58, 6, 57, 10, 53, 1], [0, 42, 55, 4, 3, 10, 38, 23, 20, 43, 60, 49, 28, 6, 34, 39, 47, 40, 27, 59, 25, 11, 36, 31, 21, 56, 48, 16, 57, 14, 1, 45, 29, 8, 2, 15, 62, 19, 44, 30, 54, 63, 58, 61, 53, 5, 22, 24, 46, 52, 41, 51, 32, 50, 17, 26, 18, 35, 37, 12, 33, 9, 7, 13], [25, 35, 2, 31, 48, 28, 47, 45, 57, 39, 50, 41, 62, 55, 8, 26, 18, 56, 60, 19, 40, 24, 38, 3, 15, 54, 53, 34, 44, 4, 36, 23, 14, 61, 10, 52, 37, 17, 30, 29, 42, 59, 12, 21, 11, 63, 16, 33, 6, 0, 32, 13, 22, 9, 1, 46, 20, 49, 51, 58, 43, 7, 27, 5], [17, 28, 63, 3, 62, 25, 6, 39, 54, 44, 34, 58, 51, 24, 40, 53, 52, 31, 5, 48, 22, 4, 16, 59, 18, 19, 20, 35, 9, 23, 41, 0, 56, 8, 14, 49, 27, 30, 13, 38, 29, 37, 26, 2, 7, 55, 33, 46, 12, 47, 21, 42, 60, 36, 43, 1, 10, 11, 50, 61, 32, 15, 45, 57], [42, 60, 3, 2, 1, 56, 25, 49, 40, 33, 13, 14, 55, 11, 54, 7, 24, 43, 31, 10, 12, 34, 21, 38, 52, 53, 22, 37, 45, 58, 44, 41, 50, 20, 32, 19, 39, 36, 30, 15, 46, 17, 26, 59, 5, 8, 28, 48, 27, 6, 63, 0, 57, 29, 62, 18, 61, 4, 47, 51, 35, 16, 9, 23], [41, 10, 55, 16, 34, 25, 17, 27, 33, 24, 43, 26, 35, 40, 3, 9, 13, 19, 52, 62, 53, 48, 15, 18, 11, 56, 60, 23, 42, 47, 22, 54, 39, 36, 31, 58, 28, 38, 37, 44, 46, 1, 14, 8, 0, 51, 5, 7, 29, 12, 4, 21, 45, 32, 2, 30, 50, 61, 59, 49, 57, 6, 20, 63], [16, 59, 28, 7, 39, 0, 37, 47, 25, 54, 23, 8, 34, 61, 14, 33, 30, 12, 2, 24, 56, 38, 31, 11, 20, 32, 50, 29, 63, 41, 19, 10, 21, 4, 35, 9, 5, 27, 48, 46, 1, 40, 52, 44, 6, 43, 18, 57, 53, 58, 26, 42, 15, 45, 17, 55, 13, 60, 3, 49, 62, 51, 36, 22], [21, 55, 27, 57, 0, 29, 14, 60, 63, 17, 52, 35, 40, 46, 24, 34, 38, 9, 42, 26, 51, 49, 33, 4, 18, 16, 44, 11, 53, 13, 31, 3, 48, 37, 45, 8, 54, 58, 22, 6, 39, 23, 32, 10, 36, 12, 15, 2, 30, 19, 61, 20, 59, 1, 56, 41, 50, 62, 5, 25, 28, 7, 43, 47], [36, 42, 2, 54, 44, 5, 28, 41, 63, 23, 14, 40, 9, 22, 15, 16, 11, 48, 49, 32, 26, 0, 13, 56, 58, 3, 38, 12, 27, 29, 39, 45, 50, 35, 21, 8, 1, 53, 37, 6, 30, 20, 33, 59, 31, 17, 7, 24, 47, 61, 25, 55, 10, 19, 62, 60, 34, 52, 46, 4, 51, 18, 57, 43], [25, 2, 57, 10, 61, 38, 31, 29, 30, 33, 19, 26, 59, 54, 28, 39, 22, 0, 35, 60, 7, 3, 21, 17, 52, 44, 48, 9, 16, 15, 42, 63, 18, 14, 37, 53, 55, 50, 13, 4, 23, 8, 32, 12, 40, 5, 24, 49, 27, 20, 45, 34, 1, 11, 46, 36, 43, 56, 58, 41, 47, 51, 62, 6], [41, 1, 5, 37, 29, 30, 0, 27, 61, 33, 48, 17, 36, 12, 32, 49, 19, 22, 55, 2, 14, 59, 8, 58, 39, 60, 25, 7, 43, 16, 53, 54, 62, 63, 40, 38, 4, 3, 46, 18, 10, 23, 45, 28, 21, 15, 24, 51, 11, 34, 56, 42, 50, 44, 13, 9, 26, 20, 52, 47, 35, 57, 6, 31], [63, 60, 54, 46, 23, 25, 56, 10, 52, 57, 59, 16, 53, 14, 26, 33, 38, 61, 22, 27, 29, 0, 58, 17, 36, 9, 21, 19, 31, 4, 34, 41, 55, 30, 40, 37, 44, 49, 32, 13, 15, 8, 50, 5, 18, 62, 47, 28, 48, 12, 11, 7, 51, 42, 43, 1, 39, 35, 6, 45, 2, 20, 24, 3], [22, 47, 5, 49, 24, 15, 13, 55, 46, 43, 17, 27, 32, 59, 31, 58, 61, 42, 20, 34, 16, 45, 2, 44, 56, 6, 50, 57, 18, 7, 4, 63, 37, 26, 1, 39, 11, 25, 29, 8, 23, 51, 0, 9, 60, 53, 54, 21, 14, 40, 48, 62, 41, 52, 33, 28, 30, 3, 38, 10, 36, 35, 19, 12], [42, 45, 34, 43, 41, 23, 55, 59, 16, 25, 44, 39, 46, 14, 50, 27, 11, 17, 36, 8, 40, 10, 15, 54, 2, 60, 33, 38, 31, 12, 48, 9, 35, 6, 24, 22, 20, 51, 13, 7, 29, 32, 47, 63, 1, 49, 19, 21, 28, 5, 18, 58, 37, 56, 4, 61, 57, 53, 26, 62, 3, 52, 0, 30], [8, 26, 49, 43, 55, 61, 52, 39, 27, 33, 18, 57, 15, 16, 30, 53, 5, 0, 36, 22, 62, 42, 51, 35, 59, 45, 50, 21, 28, 34, 13, 63, 38, 47, 37, 17, 14, 11, 46, 56, 58, 48, 32, 10, 3, 31, 29, 9, 40, 44, 23, 54, 4, 12, 19, 1, 41, 60, 2, 25, 20, 6, 24, 7], [32, 31, 26, 37, 17, 56, 5, 29, 9, 28, 41, 35, 6, 46, 36, 7, 18, 22, 23, 4, 2, 3, 24, 25, 61, 58, 30, 34, 43, 11, 15, 59, 39, 55, 21, 19, 57, 62, 40, 54, 50, 33, 1, 45, 13, 12, 42, 53, 47, 60, 0, 8, 48, 63, 52, 14, 38, 49, 27, 20, 51, 16, 10, 44], [49, 52, 31, 3, 37, 30, 46, 39, 0, 62, 42, 47, 9, 1, 4, 34, 43, 63, 59, 11, 57, 6, 20, 54, 19, 25, 8, 21, 32, 40, 23, 22, 55, 16, 13, 38, 61, 36, 29, 44, 33, 50, 18, 10, 35, 51, 14, 45, 2, 53, 17, 12, 24, 7, 28, 56, 5, 26, 60, 27, 58, 48, 41, 15], [4, 29, 48, 15, 26, 39, 57, 9, 49, 41, 17, 63, 35, 0, 16, 32, 6, 56, 2, 55, 51, 3, 22, 31, 47, 46, 33, 27, 53, 61, 40, 52, 10, 42, 45, 43, 58, 13, 8, 24, 1, 36, 18, 19, 23, 50, 38, 62, 25, 12, 54, 59, 14, 34, 11, 20, 5, 37, 21, 28, 60, 44, 30, 7], [10, 62, 57, 12, 23, 14, 27, 18, 51, 20, 2, 59, 31, 53, 19, 47, 42, 29, 6, 32, 49, 5, 3, 41, 15, 46, 9, 30, 1, 4, 34, 13, 43, 52, 48, 28, 17, 54, 24, 39, 25, 11, 37, 45, 7, 38, 0, 36, 33, 26, 55, 50, 63, 58, 35, 40, 56, 60, 44, 21, 61, 8, 22, 16], [0, 49, 29, 41, 53, 3, 10, 34, 33, 47, 51, 4, 62, 16, 19, 23, 27, 36, 28, 61, 38, 45, 6, 50, 37, 18, 11, 54, 13, 1, 31, 12, 26, 2, 48, 42, 14, 17, 63, 25, 55, 58, 46, 35, 8, 43, 7, 22, 60, 39, 5, 57, 56, 15, 21, 30, 52, 40, 20, 59, 32, 24, 44, 9], [8, 23, 56, 50, 39, 17, 49, 58, 18, 2, 33, 38, 11, 19, 16, 48, 42, 0, 36, 9, 53, 60, 5, 40, 46, 12, 25, 31, 3, 35, 21, 1, 47, 15, 51, 54, 34, 4, 44, 6, 55, 30, 41, 57, 43, 14, 20, 7, 61, 22, 13, 59, 52, 62, 28, 32, 29, 10, 45, 63, 26, 24, 37, 27], [15, 48, 43, 26, 5, 24, 38, 42, 19, 32, 4, 37, 16, 62, 33, 22, 44, 61, 40, 56, 59, 63, 58, 60, 35, 7, 8, 49, 0, 45, 13, 23, 31, 34, 2, 27, 52, 6, 41, 57, 17, 9, 21, 10, 28, 3, 36, 39, 51, 55, 54, 20, 53, 46, 1, 25, 11, 18, 50, 47, 29, 30, 12, 14], [16, 10, 21, 5, 51, 34, 62, 3, 55, 35, 52, 8, 6, 61, 59, 24, 48, 37, 44, 1, 27, 18, 2, 20, 11, 47, 31, 53, 39, 42, 4, 28, 0, 60, 43, 54, 50, 9, 25, 26, 33, 13, 38, 46, 36, 63, 15, 32, 23, 17, 14, 56, 45, 49, 7, 19, 40, 22, 30, 41, 12, 29, 57, 58], [18, 40, 47, 24, 2, 28, 27, 15, 57, 7, 5, 31, 43, 61, 23, 13, 63, 29, 55, 22, 6, 53, 9, 41, 30, 33, 49, 59, 50, 51, 38, 19, 46, 3, 36, 21, 52, 10, 56, 54, 44, 42, 39, 60, 34, 14, 1, 62, 26, 11, 0, 58, 4, 20, 17, 48, 16, 12, 32, 37, 45, 35, 25, 8], [2, 3, 32, 49, 13, 60, 16, 37, 39, 61, 50, 57, 24, 8, 42, 11, 43, 54, 5, 21, 15, 45, 46, 55, 19, 14, 48, 52, 63, 30, 25, 18, 56, 33, 20, 47, 9, 38, 23, 12, 6, 58, 27, 62, 29, 31, 4, 51, 41, 28, 22, 1, 53, 36, 35, 0, 10, 59, 44, 7, 26, 40, 34, 17], [12, 59, 42, 34, 8, 44, 10, 58, 33, 4, 53, 15, 16, 14, 52, 9, 49, 56, 18, 46, 36, 55, 39, 3, 38, 61, 28, 23, 50, 7, 30, 11, 32, 21, 43, 40, 0, 5, 47, 24, 63, 17, 20, 27, 31, 48, 54, 29, 45, 41, 2, 60, 19, 6, 1, 37, 57, 35, 13, 51, 22, 62, 26, 25], [63, 10, 7, 50, 47, 35, 60, 56, 37, 43, 5, 46, 9, 36, 29, 3, 45, 32, 14, 57, 31, 55, 42, 51, 1, 30, 39, 2, 34, 24, 0, 4, 40, 25, 26, 22, 58, 27, 19, 20, 59, 28, 6, 15, 62, 16, 12, 52, 18, 17, 53, 23, 13, 48, 44, 11, 33, 54, 61, 8, 38, 49, 21, 41], [15, 57, 14, 28, 2, 43, 21, 27, 37, 40, 47, 54, 11, 36, 23, 56, 44, 34, 38, 26, 18, 19, 9, 45, 16, 7, 32, 8, 12, 31, 17, 53, 10, 13, 50, 39, 35, 29, 62, 6, 58, 52, 60, 49, 59, 25, 48, 3, 63, 0, 51, 41, 4, 30, 5, 33, 55, 42, 24, 20, 22, 61, 1, 46], [18, 42, 37, 44, 17, 9, 27, 52, 43, 35, 10, 19, 56, 28, 54, 7, 24, 30, 59, 36, 15, 31, 20, 48, 41, 4, 11, 0, 23, 53, 1, 14, 38, 12, 8, 55, 51, 60, 16, 26, 39, 61, 34, 50, 22, 45, 6, 3, 2, 57, 62, 46, 47, 5, 29, 25, 13, 63, 40, 58, 33, 32, 21, 49], [12, 37, 30, 53, 23, 29, 8, 49, 7, 3, 43, 6, 40, 55, 54, 39, 19, 44, 16, 21, 0, 38, 62, 51, 5, 2, 60, 27, 35, 48, 56, 33, 18, 50, 41, 61, 58, 25, 17, 4, 15, 47, 13, 20, 9, 11, 24, 63, 14, 28, 22, 42, 59, 26, 1, 52, 31, 57, 32, 34, 10, 45, 46, 36], [6, 9, 54, 36, 10, 46, 26, 5, 14, 47, 51, 2, 19, 20, 30, 33, 15, 23, 21, 55, 7, 57, 18, 62, 41, 61, 60, 4, 8, 39, 53, 35, 1, 40, 38, 48, 42, 32, 31, 63, 50, 22, 59, 24, 43, 28, 11, 34, 29, 56, 16, 44, 45, 12, 37, 49, 17, 25, 58, 52, 13, 27, 0, 3], [43, 11, 33, 7, 55, 34, 10, 51, 48, 3, 57, 50, 39, 61, 36, 9, 23, 63, 20, 27, 52, 56, 41, 49, 44, 0, 60, 37, 62, 25, 38, 22, 32, 18, 29, 46, 42, 45, 5, 14, 19, 31, 1, 53, 4, 24, 17, 59, 13, 28, 47, 35, 8, 12, 54, 15, 30, 6, 40, 16, 2, 26, 21, 58], [56, 52, 32, 41, 5, 61, 19, 49, 20, 28, 1, 44, 63, 22, 58, 3, 0, 47, 34, 26, 31, 60, 59, 18, 55, 24, 16, 12, 35, 23, 33, 11, 15, 48, 29, 8, 54, 51, 57, 30, 53, 4, 43, 2, 13, 62, 46, 36, 21, 27, 14, 42, 37, 39, 7, 17, 6, 10, 45, 25, 40, 38, 50, 9], [35, 29, 15, 3, 10, 37, 43, 13, 31, 49, 57, 9, 47, 55, 0, 5, 60, 12, 54, 42, 1, 33, 19, 46, 23, 11, 16, 25, 45, 30, 63, 59, 41, 40, 34, 36, 8, 21, 50, 14, 38, 27, 24, 39, 26, 58, 62, 2, 52, 7, 44, 4, 32, 48, 17, 28, 51, 56, 22, 18, 6, 61, 53, 20], [7, 6, 52, 61, 30, 59, 42, 33, 31, 5, 28, 39, 44, 12, 17, 47, 55, 51, 35, 13, 36, 29, 27, 50, 25, 46, 10, 26, 2, 15, 40, 11, 43, 9, 3, 62, 4, 0, 53, 14, 56, 34, 8, 24, 16, 60, 32, 41, 18, 57, 37, 49, 19, 45, 21, 22, 1, 54, 63, 20, 48, 38, 23, 58], [40, 43, 61, 33, 12, 14, 19, 59, 7, 60, 21, 44, 5, 4, 23, 53, 37, 52, 18, 28, 16, 27, 35, 32, 11, 30, 6, 10, 54, 51, 56, 0, 17, 50, 13, 20, 55, 2, 31, 34, 39, 42, 26, 49, 38, 46, 62, 58, 47, 9, 15, 36, 48, 22, 57, 63, 8, 45, 1, 29, 25, 3, 41, 24], [12, 5, 57, 25, 4, 47, 36, 11, 59, 43, 18, 48, 6, 61, 17, 63, 24, 46, 38, 56, 45, 22, 39, 0, 31, 23, 53, 9, 27, 15, 30, 34, 55, 16, 50, 40, 13, 8, 7, 2, 42, 44, 51, 54, 28, 19, 33, 49, 10, 29, 52, 62, 41, 60, 37, 26, 14, 58, 20, 3, 21, 1, 32, 35], [21, 45, 51, 3, 12, 42, 26, 14, 16, 24, 9, 62, 38, 20, 39, 23, 5, 33, 43, 54, 57, 55, 6, 18, 4, 56, 27, 53, 17, 28, 15, 19, 10, 1, 50, 59, 60, 58, 48, 49, 47, 61, 46, 37, 36, 22, 34, 63, 30, 11, 32, 8, 44, 35, 29, 52, 31, 0, 41, 2, 40, 13, 25, 7], [38, 53, 36, 26, 24, 4, 57, 59, 43, 18, 44, 12, 61, 16, 33, 9, 22, 7, 62, 45, 17, 25, 21, 48, 3, 54, 31, 28, 34, 13, 60, 55, 47, 6, 52, 14, 51, 20, 11, 40, 5, 46, 63, 23, 27, 35, 37, 42, 1, 41, 0, 8, 50, 10, 15, 56, 30, 2, 19, 39, 32, 29, 58, 49]]
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
        ind = random.randint(0,self.M-1)
        choice_orig = self.choice[ind]
        child_id = self.lookup_table[ind][choice_orig]
        po2_child_id = self.lookup_table[ind][choice_orig^1]
        score_other = self.scores[po2_child_id]
        _node = nodes['{}{:d}'.format(self.child_prefix,child_id)]
        score,choice = _node.update_score(score_other,choice_orig,ts)
        self.scores[child_id] = score
        self.choice[ind] = choice

        if self.debug > 1:
            print("n_flow_on chosen {} out of -".format(child_id))
        
        return child_id, bucket_id


class ReservoirDistributionBuffer(SamplingBuffer):
        '''
        @brief:
            A simple reservoir buffer that statelessly tracks the distribution of flows across servers.
            The size of this buffer should be adaptive to:
            i) number of working servers [TBD]
            ii) traffic rate [TBD]
            so that the sampled number of on-going flow for each server is within a
            reasonable range, e.g. (0, #cpu_per_server], and the 
        '''
        
        def __init__(self, size, p=1., fresh_base=0.9):
            super().__init__(size, p, fresh_base)
            
            # initialize values as -1
            self.values = -np.ones(size).astype(int)

        def value_counter(self):
            '''
            @brief: count the occurance of all values in an array
            '''
            result = Counter(self.values)
            if -1 in result.keys(): del result[-1] # remove placeholders
            return result

class NodeLBRS(NodeLB):
    '''
    @brief:
        select AS based on statistically shortest queue
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
        po2=False,
        debug=0,
        res_buffer_size=128):

        super().__init__(
            id,
            child_ids,
            bucket_size,
            weights,
            max_n_child,
            T0,
            reward_option,
            ecmp,
            child_prefix,
            debug = debug)

        self.po2 = po2
        self.res_n_flow = ReservoirDistributionBuffer(res_buffer_size)

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

            print("@nodeLBRS {} - n_flow_on: {}".format(self.id, n_flow_on))
        
        qlen_all = np.zeros(len(self.child_ids)).astype(int)
        for k, v in self.res_n_flow.value_counter().items():
            qlen_all[k] = v
        
        if self.po2:
            n_flow_on_2 = {v: qlen_all[i]
                           for i, v in random.sample(list(enumerate(self.child_ids)), 2)}
            child_id = min(n_flow_on_2, key=n_flow_on_2.get)
            if self.debug > 1:
                not_chosen_dip = list(set(n_flow_on_2.keys())-set([child_id]))[0]
                chosen_index, not_chosen_index = list(self.child_ids).index(child_id), list(self.child_ids).index(not_chosen_dip)
                print("dips=[{:2d}, {:2d}] | #flow=[{:2d}, {:2d}] | score=[{:2d}, {:2d}] | chosen {:2d} | correct: {}".format(
                child_id, not_chosen_dip, gt['n_flow'][chosen_index], gt['n_flow'][not_chosen_index], qlen_all[chosen_index], qlen_all[not_chosen_index], child_id, gt['n_flow'][chosen_index] <= gt['n_flow'][not_chosen_index]
                ))
        else:
            n_flow_map = zip(self.child_ids, qlen_all)
            min_n_flow = min(qlen_all)
            min_ids = [k for k, v in n_flow_map if v == min_n_flow]
            child_id = random.choice(min_ids)
            if self.debug > 1:
                chosen_index = list(self.child_ids).index(child_id)
                gt_sort_dip = [self.child_ids[i] for i in np.argsort(gt['n_flow'])]
                print("chosen {:2d} | actual ranking {:2d}/{:2d}] | score vs. gt=[{:2d}, {:2d}]\n>>  qlen_all={}\n>> gt_n_flow={}".format(
                child_id, gt_sort_dip.index(child_id), len(self.child_ids), qlen_all[chosen_index], gt['n_flow'][chosen_index], qlen_all, gt['n_flow']
                ))

        self.res_n_flow.put(ts, child_id)            

        return child_id, bucket_id

class NodeLBGeometry(NodeLB):
    '''
    @brief:
        select AS based on geometry information
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
        debug=0,
        slope=2, # 1/average_fct
        weighted=False,
        sed=False,
        ):

        super().__init__(
            id,
            child_ids,
            bucket_size,
            weights,
            max_n_child,
            T0,
            reward_option,
            ecmp,
            child_prefix,
            debug = debug)

        self.t_last = 0
        self.slope=slope
        self.geometry_ts = np.zeros(len(self.child_ids))
        self.geometry_n_flow = np.zeros(len(self.child_ids))
        self.weighted = weighted
        self.sed = sed
        self.weights = self.weights[self.child_ids]
        if weighted:
            self.weights /= sum(self.weights)

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

            print("@nodeLBGeometry {} - n_flow_on: {}".format(self.id, n_flow_on))
        
        if self.weighted:
            # po2_chosen_dip = [0,0]
            # while(po2_chosen_dip[0] == po2_chosen_dip[1]):
            #     indexes_table = np.random.choice(self.child_ids, 2)
            #     rand_num = np.random.choice(self.child_ids, 2)
            #     po2_chosen_dip = get_index_alias_method(indexes_table, rand_num)
            
            po2_chosen_dip = np.random.choice(self.child_ids, 2, p=self.weights, replace=False)
            po2_chosen_idx_dip = [(self.child_ids.index(dip), dip) for dip in po2_chosen_dip]
        else:
            po2_chosen_idx_dip = random.sample(list(enumerate(self.child_ids)), 2)
        n_flow_on_2 = {}
        for i, v in po2_chosen_idx_dip:
            self.geometry_n_flow[i] = max(0, self.geometry_n_flow[i] - self.slope*(ts - self.geometry_ts[i]))
            if self.sed:
                n_flow_on_2[v] = (self.geometry_n_flow[i]+1)/self.weights[i]
            else:
                n_flow_on_2[v] = self.geometry_n_flow[i]
            self.geometry_ts[i] = ts
        child_id = min(n_flow_on_2, key=n_flow_on_2.get)
        
        if True:
            not_chosen_dip = list(set(n_flow_on_2.keys())-set([child_id]))[0]
            chosen_index, not_chosen_index = list(self.child_ids).index(child_id), list(self.child_ids).index(not_chosen_dip)
            print("dips=[{:2d}, {:2d}] | #flow=[{:2d}, {:2d}] | score=[{:.3f}, {:.3f}] | chosen {:2d} | correct: {}".format(
            child_id, not_chosen_dip, gt['n_flow'][chosen_index], gt['n_flow'][not_chosen_index], self.geometry_n_flow[chosen_index], self.geometry_n_flow[not_chosen_index], child_id, gt['n_flow'][chosen_index] <= gt['n_flow'][not_chosen_index]
            ))

        child_idx = list(self.child_ids).index(child_id)
        self.geometry_n_flow[child_idx] = gt['n_flow'][child_idx] + 1
        
        return child_id, bucket_id


class NodeLBProbFlow(NodeLB):
    '''
    @brief:
        select AS based on statistically local observation of on-going flows
        based on SYN-FIN
    '''

    def __count_obs_dips(self):
        result = {i: int(0) for i in self.child_ids}
        counters = Counter(self.obs_dip)
        if -1 in counters.keys(): del counters[-1] # remove placeholders
        for k, v in counters.items():
            result[k] = v
        return result

    def __get_min_dip(self, gt=None):
        min_n_flow = min(self.n_flow_dip.values())
        min_ids = [k for k, v in self.n_flow_dip.items() if v == min_n_flow]
        child_id = random.choice(min_ids)
        if self.debug > 1:
            counters = self.__count_obs_dips()
            assert counters == self.n_flow_dip, ">> counters={}\n>> n_flow_dip={}".format(counters, self.n_flow_dip)
            chosen_index = list(self.child_ids).index(child_id)
            gt_sort_dip = [self.child_ids[i] for i in np.argsort(gt['n_flow'])]
            print("chosen {:2d} | actual ranking {:2d}/{:2d}] | score vs. gt=[{:2d}, {:2d}]\n>>  n_flow_dip={}\n>> gt_n_flow={}".format(
            child_id, gt_sort_dip.index(child_id), len(self.child_ids), self.n_flow_dip[child_id], gt['n_flow'][chosen_index], self.n_flow_dip, gt['n_flow']
            ))
        return child_id

    def __get_min_dip_po2(self, gt=None):
        if self.weighted:
            po2_chosen_dip = np.random.choice(self.child_ids, 2, p=self.weights, replace=False)
        else:
            po2_chosen_dip = random.sample(self.child_ids, 2)
        if self.n_flow_dip[po2_chosen_dip[0]] <= self.n_flow_dip[po2_chosen_dip[1]]:
            child_id = po2_chosen_dip[0]
        else:
            child_id = po2_chosen_dip[1]

        if self.debug > 1:
            counters = self.__count_obs_dips()
            assert counters == self.n_flow_dip, ">> counters={}\n>> n_flow_dip={}".format(counters, self.n_flow_dip)
            not_chosen_dip = list(set(po2_chosen_dip)-set([child_id]))[0]
            chosen_index, not_chosen_index = list(self.child_ids).index(child_id), list(self.child_ids).index(not_chosen_dip)
            print("dips=[{:2d}, {:2d}] | #flow=[{:2d}, {:2d}] | score=[{:.3f}, {:.3f}] | chosen {:2d} | correct: {}".format(
            child_id, not_chosen_dip, gt['n_flow'][chosen_index], gt['n_flow'][not_chosen_index], self.n_flow_dip[child_id], self.n_flow_dip[not_chosen_dip], child_id, gt['n_flow'][chosen_index] <= gt['n_flow'][not_chosen_index]
            ))
        return child_id

    def __get_hash_idx(self, flow_id):
        return hash_2tuple(flow_id, flow_id*2+7) % self.bucket_size_obs

    def __obs_register_flow(self, ts, flow_id, dip):
        '''
        @params:
            ts: timestamp
            flow_id: id of the flow
            dip: the chosen dip
        '''
        flow_id = int(flow_id.split('-')[-1])
        idx = self.__get_hash_idx(flow_id)
        if self.obs_ts[idx] < 0: # available
            hash0 = flow_id % self.hash0_mod
            hash1 = flow_id % self.hash1_mod
            self.obs_hash0[idx] = hash0
            self.obs_hash1[idx] = hash1
            self.obs_dip[idx] = dip
            self.obs_ts[idx] = ts
            self.n_flow_dip[dip] += 1
        else:
            self.n_collision += 1

    def __obs_expire_flow(self, flow_id):
        '''
        @params:
            flow_id: id of the flow
        '''
        flow_id = int(flow_id.split('-')[-1])
        idx = self.__get_hash_idx(flow_id)
        hash0 = flow_id % self.hash0_mod
        hash1 = flow_id % self.hash1_mod
        # only expire the entry if ts is valid, and the 2 hashes are valid
        if self.obs_ts[idx] > 0 and self.obs_hash0[idx] == hash0 and self.obs_hash1[idx] == hash1:
            self.obs_ts[idx] = -1
            dip = self.obs_dip[idx]
            self.obs_dip[idx] = -1
            self.n_flow_dip[dip] -= 1
            assert self.n_flow_dip[dip] >= 0

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
        po2=False,
        debug=0,
        bucket_size_obs=8192, # observe table bucket size
        hash0_mod=17,
        hash1_mod=29,
        weighted=False,
        ):

        super().__init__(
            id,
            child_ids,
            bucket_size,
            weights,
            max_n_child,
            T0,
            reward_option,
            ecmp,
            child_prefix,
            debug = debug)

        if po2: 
            self.select_dip = self.__get_min_dip_po2
        else:
            self.select_dip = self.__get_min_dip
        self.n_flow_dip = {i: 0 for i in self.child_ids}
        self.bucket_size_obs = bucket_size_obs
        self.obs_ts = -np.ones(bucket_size_obs) # state of the connection
        self.obs_hash0 = np.zeros(bucket_size_obs) # 1st hash to check collision
        self.obs_hash1 = np.zeros(bucket_size_obs) # 2st hash to check collision
        self.obs_dip = -np.ones(bucket_size_obs).astype(int) # actual dip that is assigned
        self.hash0_mod = hash0_mod # the mod for the 1st hash function
        self.hash1_mod = hash1_mod # the mod for the 2nd hash function
        self.obs_bucket_mask = bucket_size_obs - 1 # make sure bucket_size_obs is a po2
        self.n_collision = 0 # number of hash collision / untracked flows
        self.weighted = weighted # do we randomly select the po2 server candidates based on weights
        if weighted:
            self.weights = self.weights[self.child_ids]
            self.weights /= sum(self.weights)

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

            print("@nodeLBGeometry {} - n_flow_on: {}".format(self.id, n_flow_on))

        child_id = self.select_dip(gt)
        
        self.__obs_register_flow(ts, flow.id, child_id)

        return child_id, bucket_id

    def expire_flow(self, ts, flow_id):
        '''
        @brief:
            receiving a FIN
            update lb node when a flow is finished and expired from bucket table
        '''
        try:
            if self.debug > 1:
                print('({:.3f}s) expire flow {} from LB bucket table'.format(ts, flow_id))
            t_begin, bucket_id, child_id = self._tracked_flows[flow_id]
            del self._tracked_flows[flow_id]
        except KeyError:  # if the flow is not tracked
            if self.debug > 1:
                print('({:.3f}s) expire flow {} from LB bucket table [untracked]'.format(ts, flow_id))
            return
        self._bucket_table_avail[bucket_id] = True
        self._counters['n_flow_on'][child_id] -= 1
        fct = ts - t_begin
        assert fct > 0
        self._reservoir['fct'][child_id].put(ts, fct)

        self.__obs_expire_flow(flow_id)

    def summarize(self):
        '''
        @brief:
            reduce flows' information during one episode
        '''

        res = {}

        res['n_untracked'] = self.n_untracked_flow
        res['n_hash_collision'] = self.n_collision

        return res