import random
import time
from config.global_conf import ACTION_DIM, RENDER, LB_PERIOD, B_OFFSET, RENDER_RECEIVE, HEURISTIC_ALPHA
from common.entities import NodeLB
import numpy as np

class NodeLBLSQ(NodeLB):

    def __init__(self, id, child_ids, bucket_size=65536, weights=None, max_n_child=ACTION_DIM, T0=time.time(), reward_option=2, ecmp=False, child_prefix='as', po2=False, debug=0):
        super().__init__(id, child_ids, bucket_size, weights, max_n_child, T0, reward_option, ecmp, child_prefix, debug)
        self.po2 = po2 # power-of-2-choices


    def choose_child(self, flow):
        # we still need to generate a bucket id to store the flow
        bucket_id, _ = self._ecmp(*flow.fields, self._bucket_table, self._bucket_mask)
        n_flow_on = self._counters['n_flow_on']
        if self.debug > 1:
            print("@nodeLBLSQ {} - n_flow_on: {}".format(self.id, n_flow_on))
        # assert len(set(self.child_ids)) == len(self.child_ids)
        if self.po2:
            n_flow_on_2 = {i: n_flow_on[i] for i in random.sample(self.child_ids, 2)}
            child_id = min(n_flow_on_2, key = n_flow_on_2.get)
            if self.debug > 1:
                print("n_flow_on chosen {} out of -".format(child_id), n_flow_on_2)
        else:
            min_n_flow = n_flow_on[self.child_ids].min()
            n_flow_map = zip(self.child_ids, n_flow_on[self.child_ids])
            min_ids = [k for k, v in n_flow_map if v == min_n_flow]
            child_id = random.choice(min_ids)
            n_flow_map = zip(self.child_ids, n_flow_on[self.child_ids])
            if self.debug > 1:
                print("n_flow_on chosen minimum {} from {}".format(child_id, '|'.join(['{}: {}'.format(k,v) for k, v in n_flow_map])))
            del n_flow_map
        return child_id, bucket_id


class NodeLBSED(NodeLB):
    '''
    @brief:
        Shortest Expected Delay (SED) assigns server based on (queue_len+1)/weight.
    '''

    def __init__(self, id, child_ids, bucket_size=65536, weights=None, max_n_child=ACTION_DIM, T0=time.time(), reward_option=2, ecmp=False, child_prefix='as', po2=False, b_offset=B_OFFSET, debug=0):
        super().__init__(id, child_ids, bucket_size, weights,
                         max_n_child, T0, reward_option, ecmp, child_prefix, debug)
        self.po2 = po2  # power-of-2-choices
        self.b_offset = b_offset

    def choose_child(self, flow):
        # we still need to generate a bucket id to store the flow
        bucket_id, _ = self._ecmp(
            *flow.fields, self._bucket_table, self._bucket_mask)
        n_flow_on = self._counters['n_flow_on']
        if self.debug > 1:
            print("@nodeLBLSQ {} - n_flow_on: {}".format(self.id, n_flow_on))
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
            
            min_n_flow = min(score)
            n_flow_map = zip(self.child_ids, score)
            min_ids = [k for k, v in n_flow_map if v == min_n_flow]
            child_id = random.choice(min_ids)
            if self.debug > 1:
                n_flow_map = zip(self.child_ids, score)
                print("score chosen minimum {} from {}".format(
                    child_id, '|'.join(['{}: {}'.format(k, v) for k, v in n_flow_map])))
            del n_flow_map
        return child_id, bucket_id

class NodeLBSRT(NodeLB):
    '''
    @brief:
        Shortest remaining time (SRT) assigns AS based on sum(cpu_processing_time)/#cpu + sum(io_processing_time)/#io
    '''

    def __init__(self, id, child_ids, bucket_size=65536, weights=None, max_n_child=ACTION_DIM, T0=time.time(), reward_option=2, ecmp=False, child_prefix='as', po2=False, debug=0):
        super().__init__(id, child_ids, bucket_size, weights,
                         max_n_child, T0, reward_option, ecmp, child_prefix, debug)
        self.po2 = po2

    def choose_child(self, flow, t_rest_all):
        
        # we still need to generate a bucket id to store the flow
        bucket_id, _ = self._ecmp(
            *flow.fields, self._bucket_table, self._bucket_mask)
        n_flow_on = self._counters['n_flow_on']
        if self.debug > 1:
            print("@nodeLBOracle {} - n_flow_on: {}".format(self.id, n_flow_on))
        # assert len(set(self.child_ids)) == len(self.child_ids) 
        t_rest_map = zip(self.child_ids, t_rest_all)
        
        if self.po2:
            t_rest_2 = {i: t_rest_all[i] for i in random.sample(self.child_ids, 2)}
            child_id = min(t_rest_2, key=t_rest_2.get)
            if self.debug > 1:
                print("n_flow_on chosen {} out of -".format(child_id), t_rest_2)
        else:
            min_t_rest = min(t_rest_all)
            min_ids = [k for k, v in t_rest_map if v == min_t_rest]
            child_id = random.choice(min_ids)
        if self.debug > 1:
            print("t_rest chosen minimum {} from {}".format(
                child_id, '|'.join(['{}: {}'.format(k, v) for k, v in t_rest_map])))
        del t_rest_map
        return child_id, bucket_id

    def receive(self, ts, flow, nodes):
        '''
        @brief:
            data plane implementation
        '''
        assert flow.nexthop == self.id
        flow.update_receive(ts, self.id)

        # select based on actual 
        t_rest_all = [nodes['{}{:d}'.format(self.child_prefix, i)].get_t_rest_total(ts)
                        for i in self.child_ids]
        child_id, bucket_id = self.choose_child(flow, t_rest_all)

        # flow = self.evaluate_decision_ground_truth(nodes, child_id, flow)
        if RENDER_RECEIVE:
            self.render_receive(ts, flow, child_id, nodes)

        # bucket is available, register flow
        if self._bucket_table_avail[bucket_id]:
            # register t_receive and chosen AS id]
            self._tracked_flows[flow.id] = (ts, bucket_id, child_id)
            self._counters['n_flow_on'][child_id] += 1
            if self.debug > 1:
                print(
                    "bucket {} available, tracking flow {} -> node {}".format(bucket_id, flow.id, child_id))
                print('n_flow_on becomes',
                      self._counters['n_flow_on'][self.child_ids])
        else:
            if self.debug > 1:
                print("bucket is not available, making flow untracked")
            self.n_untracked_flow += 1

        ts += self.get_process_delay()  # add process delay
        # for now, we only implement for ecmp_random
        flow.update_send(ts, '{}{}'.format(self.child_prefix, child_id))
        self.send(ts+self.get_t2neighbour(), flow)

        nodes['{}{}'.format(self.child_prefix, child_id)
              ].update_pending_fct(flow)


class NodeLBGSQ(NodeLB):
    '''
    @brief:
        select AS based on global shortest queue
    '''

    def __init__(self, id, child_ids, bucket_size=65536, weights=None, max_n_child=ACTION_DIM, T0=time.time(), reward_option=2, ecmp=False, child_prefix='as', po2=False, debug=0):
        super().__init__(id, child_ids, bucket_size, weights,
                         max_n_child, T0, reward_option, ecmp, child_prefix, debug)

        self.po2 = po2

    def choose_child(self, flow, qlen_all):
        
        # we still need to generate a bucket id to store the flow
        bucket_id, _ = self._ecmp(
            *flow.fields, self._bucket_table, self._bucket_mask)
        if self.debug > 1:
            print("@nodeLBOracle {} - n_flow_on: {}".format(self.id, n_flow_on))
        n_flow_map = zip(self.child_ids, qlen_all)
        if self.po2:
            n_flow_on_2 = {i: qlen_all[i]
                           for i in random.sample(self.child_ids, 2)}
            child_id = min(n_flow_on_2, key=n_flow_on_2.get)
            if self.debug > 1:
                print("n_flow_on chosen {} out of -".format(child_id), n_flow_on_2)
        else:
            min_n_flow = min(qlen_all)
            min_ids = [k for k, v in n_flow_map if v == min_n_flow]
            child_id = random.choice(min_ids)
            if self.debug > 1:
                print("n_flow_on chosen minimum {} from {}".format(
                    child_id, '|'.join(['{}: {}'.format(k, v) for k, v in n_flow_map])))
        del n_flow_map
        return child_id, bucket_id

    def receive(self, ts, flow, nodes):
        '''
        @brief:
            data plane implementation
        '''
        assert flow.nexthop == self.id
        flow.update_receive(ts, self.id)

        # select based on actual 
        qlen_all = [nodes['{}{:d}'.format(self.child_prefix, i)].get_n_flow_on()
                        for i in self.child_ids]       
        child_id, bucket_id = self.choose_child(flow, qlen_all)

        # flow = self.evaluate_decision_ground_truth(nodes, child_id, flow)
        if RENDER_RECEIVE:
            self.render_receive(ts, flow, child_id, nodes)

        # bucket is available, register flow
        if self._bucket_table_avail[bucket_id]:
            # register t_receive and chosen AS id]
            self._tracked_flows[flow.id] = (ts, bucket_id, child_id)
            self._counters['n_flow_on'][child_id] += 1
            if self.debug > 1:
                print(
                    "bucket {} available, tracking flow {} -> node {}".format(bucket_id, flow.id, child_id))
                print('n_flow_on becomes',
                      self._counters['n_flow_on'][self.child_ids])
        else:
            if self.debug > 1:
                print("bucket is not available, making flow untracked")
            self.n_untracked_flow += 1

        ts += self.get_process_delay()  # add process delay
        # for now, we only implement for ecmp_random
        flow.update_send(ts, '{}{}'.format(self.child_prefix, child_id))
        self.send(ts+self.get_t2neighbour(), flow)

        nodes['{}{}'.format(self.child_prefix, child_id)
              ].update_pending_fct(flow)


class NodeLBActive(NodeLB):

    def __init__(self, id, child_ids, bucket_size=65536, weights=None, max_n_child=ACTION_DIM, T0=time.time(), reward_option=2, ecmp=False, child_prefix='as', lb_period=LB_PERIOD, rtt_min=0.05, rtt_max=0.2, debug=0):
        super().__init__(id, child_ids, bucket_size, weights,
                         max_n_child, T0, reward_option, ecmp, child_prefix, debug, lb_period)

        self.alpha = HEURISTIC_ALPHA
        self.rtt_min = rtt_min
        self.rtt_max = rtt_max
        self.lb_period = lb_period

        assert 0 < self.alpha <= 1
    
    def get_process_delay(self):
        return random.uniform(self.rtt_min, self.rtt_max)


    def step(self, ts, nodes=None):
        '''
        @brief:
            calculate weights based on latest observation (number of on-going)
        '''
        # step 1: prediction
        qlen_all = np.array([nodes['{}{:d}'.format(self.child_prefix, i)].get_n_flow_on()
                        for i in self.child_ids])
        new_weights = np.zeros(self.max_n_child)
        new_weights[self.child_ids] = max(qlen_all) - qlen_all
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
        self.register_event(ts + self.lb_period,
                            'lb_step', {'node_id': self.id})
