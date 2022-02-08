# ---------------------------------------------------------------------------- #
#                                  Description                                 #
# This file defines entities (or components) in simulators, including:         #
#   - Event: basic building block in the event-driven simulator that happens   #
#            at a given timestamp                                              #
#   - Flow:  query sent by client / job to be processed by application server  #
#   - FlowBuffer: a buffer that helps process & digest flow info               #
#   - WorkerQueue: an FIFO queue that stores flows under process               #
#   - Node: an abstract class for all nodes                                    #
#   - NodeAS: application server node                                          #
#   - NodeLB: load balancer node                                               #
# ---------------------------------------------------------------------------- #

import time
from inspect import getfullargspec
from collections import namedtuple
from heapq import heappush, heappop, heapify
from config.global_conf import *
from common.utils import *

Event = namedtuple('Event', 'time name added_by kwargs')
Gaussian = namedtuple('Gaussian', ['mean', 'var'])
Gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])

class Flow:
    '''
    @brief:
        Flows are stored in tuple format and this simple implementation of network 
        flow class helps interpret the tuple and update given fields under various events.
    '''

    def __init__(self, id, ts, nexthop, fct, fct_type='cpu', fields=[]):
        self.id = id
        self.src_node = self.id.split('-')[0]
        self.t_begin = ts
        if isinstance(fct, float): fct = [fct]
        self.fct = fct
        self.fct_index = 0
        self.fct_type = fct_type # 'cpu' or 'io'
        self.t_rest = fct[0]
        self.t_end = None
        self.nexthop = nexthop
        self.path = []
        self.tss = []
        self.fields = fields # we could put 2/5-tuple info here
        self.note = {}

    def get_info(self):
        '''·
        @brief:
            generate information of this flow
        '''

        info = 'Flow ({}) | {}[{:.3f}]'.format(
            self.id, self.src_node, self.t_begin)
        i = 0
        while i < len(self.tss):
            if i % 2:
                info += '[{:.3f}]'.format(self.tss[i])
            else:
                info += '->[{:.3f}]{}'.format(self.tss[i], self.path[int(i/2)])
            i += 1
        if self.t_end: 
            # check timestamps
            assert self.t_end > self.t_begin
            assert len(self.tss) == 2*len(self.path)
            assert self.tss == sorted(self.tss), "weird timestamps w/ tss {} throughout path {}".format(self.tss, self.path)
            assert self.t_end > max(
                self.tss), "Flow {}: weird timestamps w/ tss {} throughout path {}, and t_end {:.9f}s".format(self.id, self.tss, self.path, self.t_end)
            assert self.t_begin < min(self.tss)
            info += '->[{:.3f}]{} (FCT={:.3f}s, PLT={:.3f}s)'.format(self.t_end,
                                                        self.src_node, sum(self.fct), self.t_end - self.t_begin)
            if 'reject' in self.nexthop:
                info += ' (rejected)'

            info += " | " + str(self.note)
            
        else:
            info += ' (Total FCT={:.3f}s, total stage={}, FCTs={}, FCT_index={}, current type={}, t_rest={:.3f}s, nexthop={})'.format(sum(self.fct), len(self.fct), self.fct, self.fct_index, self.fct_type, self.t_rest, self.nexthop)
        return info

    def print_info(self):
        '''
        @brief:
            print out flow information
        '''
        print(self.get_info())

    def update_receive(self, ts, receiver_id, terminal=False):
        '''
        @params:
            ts: timestamp
            receiver_id: id of the receiver node
            terminal: if the flow has returned to the src_node
        '''
        if self.nexthop == receiver_id:
            if terminal:
                assert self.src_node in self.nexthop
                self.t_end = ts                
            else:
                self.tss.append(ts)
                self.path.append(receiver_id)
            return 0  # normal case
        else: # in case of just finished being processed
            assert receiver_id in self.nexthop, "{}: receiver_id {} not in nexthop {}".format(self.get_info(), receiver_id, self.nexthop)
            if terminal:
                self.t_end = ts+T_TIMEOUT
            else:
                self.nexthop = receiver_id
            return 1

    def update_send(self, ts, nexthop):
        '''
        @params:
            ts: timestamp
            nexthop: next node to receive
        @note:
            client node do not send since it registers all flows during initialization phase
        '''
        self.tss.append(ts)
        self.nexthop = nexthop
        
    def update_fct_stage(self):
        '''
        @brief:
            move the flow to the next process stage if it existed 
        '''
        self.fct_index += 1
        if self.fct_index >= len(self.fct): return -1
        if self.fct_type == 'cpu': self.fct_type = 'io'
        else: self.fct_type = 'cpu'
        self.t_rest = self.fct[self.fct_index]
        return 0

    def get_plt(self):
        assert self.t_end, "weird t_end of flow {}".format(self.get_info())
        return self.t_end - self.t_begin

    def get_fct(self):
        assert self.t_end, "weird t_end of flow {}".format(self.get_info())
        return self.tss[-1] - self.tss[-2]

    def get_fct_expected(self):
        return sum(self.fct)

class DistributionFCT(object):

    def __init__(self, fct_type, **kwargs):
        all_types_fct = ["exp", "normal", "uniform", "lognormal"]
        fct_type_kwargs = {
            "exp": ["mu"],
            "normal": ["mu", "std"],
            "uniform": ["mu", "std"],
            "lognormal": ["mu", "std"]
        }

        assert fct_type in all_types_fct
        self.fct_type = fct_type
        self.generator = eval("get_fct_{}".format(fct_type))        
        self.kwargs_keys = fct_type_kwargs[fct_type]
        assert set(self.kwargs_keys) == set(
            kwargs.keys()), "expected fct characteristics are: {}".format(self.kwargs_keys) + ", but get {}".format(kwargs)
        self.kwargs = kwargs

    def get_value(self):
        return self.generator(*[self.kwargs[k] for k in self.kwargs_keys])


class Application(object):
    
    def __init__(self, rate, n_stage, cpu_distribution, io_distribution, first_stage='cpu'):
        '''
        @brief:
            implement one type of application that generates flows with a consistent characteristics
        '''
        self.update_config(rate, n_stage, cpu_distribution, io_distribution, first_stage)

    def update_config(self, rate=None, n_stage=None, cpu_distribution=None, io_distribution=None, first_stage=None, **kwargs):
        if rate:
            assert rate > 0
            self.rate = rate
        if n_stage:
            assert n_stage > 0 and isinstance(n_stage, int)
            self.n_stage = n_stage
        if cpu_distribution:
            self.cpu_fct = DistributionFCT(**cpu_distribution)
        if io_distribution:
            self.io_fct = DistributionFCT(**io_distribution)
        if first_stage:
            assert first_stage in ['cpu', 'io']
            self.first_stage = first_stage
            n_stage_half = int(self.n_stage/2)
            if first_stage == 'cpu':
                self.cpu_fct_idx = range(0, self.n_stage, 2)
                self.io_fct_idx = range(1, self.n_stage, 2)
                self.n_stage_cpu = self.n_stage - n_stage_half
                self.n_stage_io = n_stage_half
            else:
                self.cpu_fct_idx = range(1, self.n_stage, 2)
                self.io_fct_idx = range(0, self.n_stage, 2)
                self.n_stage_cpu = n_stage_half
                self.n_stage_io = self.n_stage - n_stage_half

    def get_query(self):
        '''
        @return:    
            fct: a list of fct's corresponding to each stage of the processing, interchanging between cpu and io;
            first_stage: the first stage type, either 'cpu' or 'io'
            dt_next: interval to the next flow to be dispatched
        '''
        fct = np.zeros(self.n_stage)
        if self.n_stage_cpu > 0:
            cpu_fct = self.cpu_fct.get_value()
            cpu_fct_pieces = number_split_n_pieces(cpu_fct, self.n_stage_cpu)
            fct[self.cpu_fct_idx] = cpu_fct_pieces
        if self.n_stage_io > 0:
            io_fct = self.io_fct.get_value()
            io_fct_pieces = number_split_n_pieces(io_fct, self.n_stage_io)
            fct[self.io_fct_idx] = io_fct_pieces
        dt_next = expovariate(self.rate)
        return fct, self.first_stage, dt_next

class PriorityQueue:
    '''
    @brief:
        a simple implementation of priority queue with heapq, (single-threaded simulator does't need synchronization)
    '''
    def __init__(self, queue_len=None):
        if queue_len:
            assert queue_len > 0 and isinstance(queue_len, int)
        self.queue_len = queue_len
        self.queue = []
        heapify(self.queue)

    def qsize(self):
        return len(self.queue)

    def empty(self):
        return self.qsize() == 0

    def full(self, qsize=None):
        res = False
        if self.queue_len:
            if qsize:
                res = qsize >= self.queue_len
            else:
                res = self.qsize() >= self.queue_len
        return res

    def put(self, element, checkfull=True):
        '''
        @note: element should have at least 2 fields, i.e. key and value
        '''
        if checkfull: assert not self.full()
        heappush(self.queue, element)

    def pop(self):
        assert not self.empty()
        return self.pop_n(1)[0]

    def remove(self, element):
        self.queue.remove(element)

    def update_len(self, len):
        self.queue_len = len

    def print_queue(self):
        for element in self.queue:
            info = ">> ({:.3f}s) ".format(element[0])
            if isinstance(element, Flow):
                info += element.get_info()
            elif isinstance(element, Event):
                event = element[1]
                if event == 'dp_receive':
                    info = "[DP] " + info
                    info += 'event {} concenring {}'.format(element[1], element[-1]['flow'].get_info())
                else:
                    info = "[CP] " + info
                    info += str(element)
            elif isinstance(element[-1][-1], Flow):
                info += element[-1][-1].get_info()
            else:
                info += str(element)
            print(info)

    def peek_n(self, n, reverse=False):
        '''
        @params:
            reverse: set to True if we want the n maximal elements
        '''
        sorted_queue = sorted(self.queue)
        if reverse:
            sorted_queue = sorted_queue[::-1]
        return sorted_queue[:min(n, len(sorted_queue))]

    def pop_n(self, n, reverse=False):
        '''
        @params:
            reverse: set to True if we want the n maximal elements
        '''
        self.queue = sorted(self.queue)
        if reverse:
            self.queue = self.queue[::-1]
        index = min(n, self.qsize())
        res = self.queue[:index]
        self.queue = self.queue[index:]
        return res

    def reset(self):
        del self.queue
        self.queue = []

class ReservoirSamplingBuffer(object):
    '''
    @brief:
        A simple implementation of reservoir sampling buffer which consists of a list of (ts, value)
    '''

    def __init__(self, size, p=1., fresh_base=0.9):
        assert size > 0
        self.size = size # buffer size
        self.table_mask = size-1
        assert 0 < p <= 1
        self.p = p
        assert 0 < fresh_base <= 1
        self.fresh_base = fresh_base
        self.tss = np.zeros(size) # timestamps
        self.values = np.zeros(size)  # values

    def __get_freshness(self, ts):
        '''
        @brief:
            calculate datapoint freshness based on timestamps
        @params:
            ts: current timestamp to compare freshness
        '''
        assert ts > max(self.tss), "ts {:.3f}s".format(ts) + str(self.tss)
        delta_t = [ts - t if t != 0 else 0 for t in self.tss]
        return np.power(self.fresh_base, delta_t)

    def put(self, ts, value):
        # with P=p we store this new sample
        if random.random() > self.p: return

        # newly added sample should always be fresher than registered samples
        assert ts > max(self.tss)
        
        index = random.randint(0, self.table_mask) # get index to store

        # store ts-1e-9 so that no assertion is triggered when several samples are stored at the same time
        self.tss[index] = ts-1e-9 
        self.values[index] = value

    def get_value_variance(self):
        return self.values.std()**2

    def summary(self, ts):
        res = {}
        assert ts >= max(self.tss), "ts={:.3f}s, max(tss)={:.3f}s".format(ts, max(self.tss))
        res['avg'] = self.values.mean()
        res['std'] = self.values.std()
        res['p90'] = np.percentile(self.values, 90)
        freshness = self.__get_freshness(ts)
        values_discounted = np.multiply(self.values, freshness)
        res['avg_disc'] = values_discounted.sum() / freshness.sum()
        res['avg_decay'] = values_discounted.mean()
        return res

    def get_info(self):
        res = ["Buffer size: {}".format(self.size)]
        res.append('{:<8s}'.format('Times:') +
              ' |'.join(["{: > 7.3f}".format(t_) for t_ in self.tss]))
        res.append('{:<8s}'.format('Values:') +
              ' |'.join(["{: > 7.3f}".format(t_) for t_ in self.tss]))
        return '\n'.join(res)

class Node(object):
    '''
    @brief:
        An abstract class implementing one node
    '''

    global event_buffer

    def __init__(self, id, debug=0):
        self.id = id
        self.debug = debug

    def get_process_delay(self):
        return random.uniform(1e-6, 1e-5)

    def get_t2neighbour(self):
        return random.uniform(1e-4, 1e-3)

    def receive(self):
        '''
        @brief:
            different nodes have different receive implementation
        '''
        raise NotImplementedError

    def register_event(self, ts, event, kwargs={}):
        '''
        @brief:
            push an Event tuple into control plane buffer
        '''
        event_buffer.put(Event(ts, event, self.id, kwargs), checkfull=False)

    def send(self, ts, flow):
        '''
        @brief:
            push the tuple of concerned flow and the timestamp when it will be received
        '''
        self.register_event(ts, 'dp_receive', {'flow': flow})

    def remove_event(self, ts, event, kwargs={}):
        '''
        @brief:
            push an Event tuple into control plane buffer
        '''
        event_buffer.remove(Event(ts, event, self.id, kwargs))

    def revoke(self, ts, flow):
        '''
        @brief:
            push the tuple of concerned flow and the timestamp when it will be received
        '''
        self.remove_event(ts, 'dp_receive', {'flow': flow})

class NodeAS(Node):
    '''
    @brief:
        An application server node with two FIFO queues (waiting & processing)
    '''

    def __init__(self, id, n_worker=N_WORKER_BASELINE, multiprocess_level=1, max_client=AS_MAX_CLIENT, debug=0):
        '''
        @params:
            multiprocess_level: #backlog = multiprocess_level*n_worker, FIFO if multiprocess_level=1
        '''
        super(self.__class__, self).__init__(id, debug)
        assert max_client > 0 and isinstance(max_client, int)
        self.max_client = max_client
        assert n_worker > 0 and isinstance(n_worker, int)
        self._init_n_worker = n_worker
        self.n_disk = 1
        self._init_mp_level = multiprocess_level
        self.process_type = set(['cpu', 'io']) # two different processing models
        self.queues = {}

        self.reset()
    
    def reset(self):
        del self.queues
        self.process_speed = {'cpu': 1, 'io': 1} # initialize process_speed
        self.n_worker = self._init_n_worker
        self.mp_level = self._init_mp_level

        self.queues = {}
        # unprocessed flows that are waiting for a worker
        self.queues['wait'] = PriorityQueue()
        # flows that are currently being processed
        self.queues['cpu'] = PriorityQueue(self.n_worker * self.mp_level)
        # flows that are currently io-wait
        self.queues['io'] = PriorityQueue()

        self.pending_fct = {
            'cpu': 0,
            'io': 0,
        }

    def get_n_flow_on(self):
        n_flow = [q.qsize() for k, q in self.queues.items()]
        return sum(n_flow)
    
    def update_pending_fct(self, flow, add=True):
        '''
        note: flow.fct = [fct_cpu_stage1, fct_io_stage1, fct_cpu_stage2, fct_io_stage2, ...]
        '''
        if add:
            self.pending_fct['cpu'] += sum(flow.fct[::2])
            self.pending_fct['io'] += sum(flow.fct[1::2])
        else:
            self.pending_fct['cpu'] -= sum(flow.fct[::2])
            self.pending_fct['io'] -= sum(flow.fct[1::2])

    def get_t_rest_total(self, ts, flow=None):
        '''
        @brief:
            calculate total (unit) processing time left for all applications
            - accumulate remaining CPU processing time and divide by number of worker threads
            - accumulate remaining IO processing time
        '''
        t_rest = {
            'cpu': 0, # remaining CPU time rest to process
            'io': 0, # remaining IO time rest to process
        }
        # process fcts in the queue
        for target in ['cpu', 'io', 'wait']:
            for _, flow in self.queues[target].queue:
                if isinstance(flow, Flow): # wait queue case
                    t_rest['cpu'] += sum(flow.fct[flow.fct_index::2])
                    t_rest['io'] += sum(flow.fct[(flow.fct_index+1)::2])
                else: # cpu/io queue case
                    t_begin, flow = flow # decode flow tuple values

                    # time left for the currently being processed stage
                    t_rest[target] += flow.t_rest - (ts - t_begin) * self.process_speed[target]

                    # if the flow has multiple stages, add the following stages' FCT as well
                    t_rest[target] += sum(flow.fct[(flow.fct_index+2)::2])
                    t_rest[list(self.process_type - set([target]))[0]] += sum(flow.fct[(flow.fct_index+1)::2])
        

        # process fcts in the pending 
        for target in ['cpu', 'io']:
            t_rest[target] += self.pending_fct[target]

        # if flow:
        #     t_rest['cpu'] += sum(flow.fct[::2])
        #     t_rest['io'] += sum(flow.fct[1::2])

        t_rest_all = t_rest['cpu'] / self.n_worker + t_rest['io'] # total time rest to process

        # for debug
        # print('='*10 + "as {}".format(self.id), '='*10)
        # for target in ["cpu", "io"]:
        #     print('{} queue'.format(target))
        #     self.queues[target].print_queue()
        #     print("pending_fct {}: {:.3f}s".format(target, self.pending_fct[target]))
        # print('wait queue length {}'.format(self.queues['wait'].qsize()))
        # print("t_rest_all = t_rest['cpu']({:.3f}s) / n_worker({}) + t_rest['io']({:.3f}s) = {:.3f}s".format(t_rest['cpu'], self.n_worker, t_rest['io'], t_rest_all))

        return t_rest_all

    def calcul_process_speed(self, n_flows2add=0, target='cpu'):
        n_flow = self.queues[target].qsize() + n_flows2add
        if n_flow == 0: return 1.
        if target == 'cpu':
            assert n_flow <= self.queues[target].queue_len
            return min(self.n_worker/n_flow, 1.)
        elif target == 'io':
            return self.n_disk/n_flow
        else:
            raise NotImplementedError

    def send2client(self, ts, flow, nexthop, reject=False):
        if reject:
            flow.update_send(ts+T_TIMEOUT, nexthop+'-reject')
        else:
            flow.update_send(ts, nexthop)
        if self.debug > 1:
            flow.print_info()
        self.send(ts+self.get_t2neighbour(), flow)

    def put2wait(self, flows2add=[], checkfull=False):
        '''
        @brief:
            put flow into wait queue
        @note: 
            by default we don't check if the wait queue is full here, in case of worker queue reduction, this might lead to wait queue length exceeding
        '''
        if isinstance(flows2add, Flow): flows2add = [flows2add]
        for flow in flows2add:
            self.queues['wait'].put((flow.tss[-1], flow), checkfull=checkfull)

    def put2process(self, ts, flows2add, target='cpu', checkfull=False):
        '''
        @brief:
            put flows into worker queue
        '''
        if isinstance(flows2add, Flow): flows2add = [flows2add]
        assert target in ['cpu', 'io']
        for flow in flows2add:
            t_finish = ts + flow.t_rest / self.process_speed[target]
            self.queues[target].put((t_finish, (ts, flow)), checkfull=checkfull)
            self.send(t_finish, flow)

    def deep_pop_n_flow_in_process(self, ts, n=None, reverse=False, target='cpu'):
        '''
        @brief:
            pop n flows in process (in worker queue) from both worker queue and data plane buffer
        '''
        # initialize result
        popped_flows = []
        # get the popped flows
        if n == None:
            popped_tuples = self.queues[target].queue
            self.queues[target].queue = []
        else:
            popped_tuples = self.queues[target].pop_n(n, reverse)

        if self.debug > 1:
            print(">> @deep_pop_n_flow_in_process (ts={:.3f}s, n={}, reverse={})".format(ts, n, reverse))

        for t_finish, (t_begin, flow) in popped_tuples:
            if self.debug > 1:
                print(">> previous ({:.3f}s) t_begin={:.3f}s".format(ts, t_begin), flow.get_info())
            self.revoke(t_finish, flow)
            flow.t_rest -= (ts - t_begin) * self.process_speed[target]
            assert flow.t_rest > 0
            if self.debug > 1:
                print(">> updated ({:.3f}s) t_begin={:.3f}s".format(ts, t_begin), flow.get_info())
            popped_flows.append(flow)
        return popped_flows

    def update_process_queue(self, ts, flows2add=[], update_process_speed=True, target='cpu'):
        if update_process_speed:
            n_flows2add = len(flows2add)
            process_speed_new = self.calcul_process_speed(n_flows2add, target=target)
            if process_speed_new != self.process_speed[target]: 
                if self.debug > 1:
                    print(">> process speed changed from {} to {}".format(self.process_speed[target], process_speed_new))
                flows2add += self.deep_pop_n_flow_in_process(ts, target=target)
            self.process_speed[target] = process_speed_new
        self.put2process(ts, flows2add, target)

    def receive(self, ts, flow, nodes):
        '''
        @brief:
            data plane implementation
        '''
        assert self.id in flow.nexthop , "{}: nexthop should be node itself {}".format(str(flow.get_info()), self.id)
        if flow.nexthop != flow.path[-1]: # the flow just arrives at AS node
            case_no = flow.update_receive(ts, self.id)
            if case_no == 0: # flow just got redirected from LB to AS 
                self.update_pending_fct(flow, add=False)
                if self.get_n_flow_on() > self.max_client: # reject only in case of first arrived
                    if self.debug > 1:
                        print("|| @AS {}: already serving max_client, reject!".format(self.id))
                    self.send2client(
                        ts+random.uniform(1e-6, 1e-5), # simulate a random processing time for rejected flows plus timeout
                        flow, flow.src_node, reject=True)
                    return
            if flow.fct_type == 'cpu':  # go directly to worker queues
                if self.queues['cpu'].full(): # put new flow into waiting queue
                    if self.debug > 1:
                        print('|| @AS {}: worker queue is full (len={}), put flow to wait queue | {}!'.format(
                            self.id, self.queues['cpu'].qsize(), flow.get_info()))
                    self.put2wait([flow])
                else: # directly send this flow to workers                
                    self.update_process_queue(ts, flows2add=[flow], target='cpu')
                    if self.debug > 1:
                        print('|| @AS {}: add to cpu queue (len={}) | {}!'.format(
                            self.id, self.queues['cpu'].qsize(), flow.get_info()))
            else:
                self.update_process_queue(ts, flows2add=[flow], target='io')
                if self.debug > 1:
                    print('|| @AS {}: add to io queue (len={}) | {}!'.format(self.id, self.queues['io'].qsize(), flow.get_info()))
            if self.debug > 1:
                print("|| @AS {}: queue info {}".format(
                    self.id, flow.fct_type))
                self.queues[flow.fct_type].print_queue()
        else: # the flow is already at AS node and will finish process
            queue2update_id = flow.fct_type
            if self.debug > 1:
                print('-'*10 + '{:.3f}s'.format(ts) + '-'*10)
                flow.print_info()
                print("|| @AS {}: queue info {}".format(self.id, queue2update_id))
                self.queues[queue2update_id].print_queue()
            _, (_, flow_) = self.queues[queue2update_id].pop()
            assert flow.id == flow_.id
            if flow.update_fct_stage() < 0: # flow is finished
                self.send2client(ts, flow, flow.src_node)
                if self.debug > 1:
                    print('|| @AS {}: finish process {}!'.format(self.id, flow.get_info()))
            else:
                flow.nexthop += '-{}'.format(flow.fct_type)
                if self.debug > 1:
                    print('|| @AS {}: context switch from {}, new flow {}'.format(self.id, queue2update_id, flow.get_info()))
                self.send(ts+1e-9, flow) # context switch
            flows2add = []
            update_process_speed = True
            if queue2update_id == 'cpu':
                if not self.queues['wait'].empty():
                    flows2add.append(self.queues['wait'].pop()[1])
                    update_process_speed = False
            self.update_process_queue(
                ts, flows2add=flows2add, update_process_speed=update_process_speed, target=queue2update_id)
            

    def update_capacity(self, ts, n_worker=None, multiprocess_level=None):
        '''
        @brief:
            control plane may change AS processing capacity from time to time, triggered by event 'as_update_capacity'
        '''

        worker_len_old = self.queues['cpu'].queue_len

        if n_worker:
            assert isinstance(n_worker, int) and n_worker > 0
            self.n_worker = n_worker
        if multiprocess_level:
            assert isinstance(multiprocess_level, int) and multiprocess_level > 0
            self.mp_level = multiprocess_level
        
        self.queues['cpu'].update_len(self.n_worker * self.mp_level)

        delta_queue_len = self.queues['cpu'].queue_len - worker_len_old # new - old

        if self.debug > 0:
            print(">> ({:.3f}s) @AS {} update capacity from {} to {} (delta={})".format(ts, self.id, worker_len_old, self.queues['cpu'].queue_len, delta_queue_len))

        if delta_queue_len > 0:
            # move flows from wait queue to worker queue
            flows2add = [tuple_[-1] for tuple_ in self.queues['wait'].pop_n(delta_queue_len)]
        else:
            # move flows from worker queue to wait queue before updating their t_rest
            flows2add = []
            flows2remove = self.deep_pop_n_flow_in_process(ts, -delta_queue_len, reverse=True, target='cpu') # prioritize keeping soon-to-finish flows in the queue
            self.put2wait(flows2remove)

        self.update_process_queue(ts, flows2add=flows2add)    

class NodeStatelessLB(Node):
    '''
    @brief:
        A stateless load balancer w/ ECMP
    '''

    def __init__(self, id, child_ids, bucket_size=65536, max_n_child=ACTION_DIM, T0=time.time(), ecmp=False, child_prefix='as', debug=0):
        '''
            @params:
                child_ids: a list of id number of AS which is currently active
                bucket_size: size of the bucket table
                T0: T0 of simulation time
                ecmp:
                    False: randomly pick one from bucket
                    2: use 2 tuple ecmp
                    5: use 5 tuple ecmp
            '''
        # initialize arguments
        super().__init__(id, debug)
        assert isinstance(max_n_child, int) and max_n_child > 0
        self.max_n_child = max_n_child
        assert np.array(child_ids).any() in range(
            max_n_child), "[not in range({})] child_ids {}".format(max_n_child, child_ids)
        self.bucket_size = bucket_size
        self._bucket_mask = bucket_size - 1
        self._init_weights = np.zeros(max_n_child)  # a mask that zeros out inactive AS
        self._init_weights[child_ids] = 1.
        self.T0 = T0
        self._ecmp = ecmp_methods[ecmp]
        self.child_prefix = child_prefix
        NodeStatelessLB.reset(self)


    def reset(self):
        '''
        @brief:
            reset this node by reinitializing all the parameters and buckets
        '''
        self.weights = self._init_weights
        self.child_ids = [i for i, v in enumerate(self.weights) if v > 0]
        self.generate_bucket_table()


    def generate_bucket_table(self):
        weights_ = self.weights/sum(self.weights)
        self._bucket_table = np.random.choice(
            range(self.max_n_child), self.bucket_size, p=weights_)
        if self.debug > 1:
            unique, counts = np.unique(self._bucket_table, return_counts=True)
            n_i = dict(zip(unique, counts))
            for i, n in n_i.items():
                print("{} {}: {} ({:.2%})".format(self.child_prefix, i, n, n/self.bucket_size))

    def receive(self, ts, flow, nodes):
        '''
        @brief:
            data plane implementation
        '''
        assert flow.nexthop == self.id
        flow.update_receive(ts, self.id)
        
        # random select 
        _, child_id = self._ecmp(*flow.fields, self._bucket_table, self._bucket_mask)

        ts += self.get_process_delay()
        flow.update_send(ts, '{}{}'.format(self.child_prefix, child_id)) # for now, we only implement for ecmp_random
        self.send(ts+self.get_t2neighbour(), flow)


    def add_child(self, child_id):
        # convert single as id to a list
        if not isinstance(child_id, list): child_id = [child_id]
        # child_id to add should be within legal range and beyond current active set
        assert len(set(child_id) - set(range(self.max_n_child))
                   ) == 0 and len(set(child_id) - set(self.child_ids)) == len(child_id)
        self.child_ids += child_id
        for id_ in child_id:
            self.weights[id_] = 1.

        self.generate_bucket_table()  # update bucket table

    def remove_child(self, child_id):
        if not isinstance(child_id, list): child_id = [child_id] # convert single as id to a list
        # child_id to add should be within legal AS range and current active as set
        assert len(set(child_id) - set(self.child_ids)) == 0 
        for id_ in child_id:
            self.child_ids.remove(id_)
            self.weights[id_] = 0.

        self.generate_bucket_table() # update bucket table

class NodeLB(NodeStatelessLB):
    '''
    @brief:
        A basic load balancing that does (weighted) ECMP
    '''

    def __init__(self, id, child_ids, bucket_size=LB_BUCKET_SIZE, weights=None, max_n_child=ACTION_DIM, T0=time.time(), reward_option=2, ecmp=False, child_prefix='as', debug=0, lb_period=LB_PERIOD):
        '''
        @params:
            child_ids: a list of id number of AS which is currently active
            bucket_size: size of the bucket table
            weights: a dict storing each active AS' weight, set to None if we don't do weighted ECMP
            T0: T0 of simulation time
            ecmp: 
                False: randomly pick one from bucket
                2: use 2 tuple ecmp
                5: use 5 tuple ecmp
            child_prefix: by default 'as'
        '''
        # initialize arguments
        super().__init__(id, child_ids, bucket_size, max_n_child, T0, ecmp, child_prefix, debug)
        if weights:
            assert np.array(list(weights.keys())).any() in range(
                max_n_child), 'LB {} - weights\' id should be in max_n_child range'.format(self.id)
            self._init_weights[child_ids] = [w for _, w in weights.items()]
            # normalize weights
            self._init_weights /= sum(self._init_weights)
        self.reward_fn = reward_options[reward_option]
        self.lb_period = lb_period
        NodeLB.reset(self) # to avoid recursive reset

    def reset(self):
        '''
        @brief:
            reset this node by reinitializing all the parameters and buckets
        '''
        super().reset()
        self._bucket_table_avail = np.array([True] * self.bucket_size)
        self._tracked_flows = {}
        # (accumulated) counter of number of untracked flows when a flow arrives meanwhile its corresponding bucket is not available
        self.n_untracked_flow = 0

        self._counters = {
            'n_flow_on': np.zeros(self.max_n_child)
        }
        self._reservoir = {
            # flow duration
            'fd': [ReservoirSamplingBuffer(RESERVOIR_BUFFER_SIZE) for _ in range(self.max_n_child)],
            # flow complete time
            'fct': [ReservoirSamplingBuffer(RESERVOIR_BUFFER_SIZE) for _ in range(self.max_n_child)],
        }
        self._res_max_ts = 0.

        if self.debug > 1:
            print("in NodeLB kickoff node {}".format(self.id))
        self.register_event(random.uniform(1e-6, 1e-5), 'lb_step', {'node_id': self.id}) # kickoff
        
    def __update_res_fd(self, ts):
        '''
        @brief:
            update flow duration reservoir buffer
        '''
        assert self._res_max_ts <= ts, "current ts {:.7f}s, res_max_ts {:.7f}s".format(
            ts, self._res_max_ts)
        # make ts lower bound larger than all samples in reservoir buffer
        self._res_max_ts += 1e-9
        if self.debug > 2:
            print('=== ({:.3f}s) LB {}: before update reservoir {} ==='.format(
                ts, self.id, 'fd'))
            for i in self.child_ids:
                print('--- {} {} ---'.format(self.child_prefix, i))
                print(self._reservoir['fd'][i].get_info())

        tv_pairs = []
        for t_begin, _, child_id in self._tracked_flows.values():
            tss_ = np.random.uniform(self._res_max_ts, ts, random.randint(
                0, RESERVOIR_FD_PACKET_DENSITY))  # randomly sample several timestamps
            values_ = [t_ - t_begin for t_ in tss_]
            for t_, v_ in zip(tss_, values_):
                if v_ > 1e-2:
                    heappush(tv_pairs, (t_, (v_, child_id)))
        while len(tv_pairs) > 0:
            if self.debug > 2:
                print("current ts {:.7f}s, res_max_ts {:.7f}s".format(
                    ts, self._res_max_ts))
                print(">> tv_pairs:", tv_pairs)
            t_, (v_, child_id) = heappop(tv_pairs)
            self._reservoir['fd'][child_id].put(t_, v_)
        try:
            self._res_max_ts = t_
        except NameError:
            if self.debug > 2:
                print('=== ({:.3f}s) LB {}: after update reservoir {} ==='.format(
                    ts, self.id, 'fd'))
                for i in self.child_ids:
                    print('--- {} {} ---'.format(self.child_prefix, i))
                    print(self._reservoir['fd'][i].get_info())

    def get_observation(self, ts):

        # initialization
        res = self._counters

        for k, v in self._reservoir.items():
            summaries = []
            for i in range(self.max_n_child):
                summaries.append(v[i].summary(ts))
            for kk in RESERVOIR_SUMMARY_KEYS:
                res.update({'res_{}_{}'.format(k, kk): np.array([
                           summaries[i][kk] for i in range(self.max_n_child)])})
        return res

    def choose_child(self, flow, nodes=None, ts=None):
        # random select 
        bucket_id, child_id = self._ecmp(*flow.fields, self._bucket_table, self._bucket_mask)
        return child_id, bucket_id

    def get_ground_truth(self, nodes, ts, flow):
        gt = {
            "n_flow": [nodes["{}{}".format(self.child_prefix, i)].get_n_flow_on() for i in self.child_ids],
            "t_remain": [nodes["{}{}".format(self.child_prefix, i)].get_t_rest_total(ts, flow) for i in self.child_ids]
        }
        return gt

    def evaluate_decision_ground_truth(self, nodes, chosen_id, flow):
        flow_on_ground_truth = {i: nodes["{}{}".format(self.child_prefix, i)].get_n_flow_on() for i in self.child_ids}
        n_flow_on_min = min(flow_on_ground_truth.values()) # minimum #flow-on across all AS
        n_flow_on_avg = np.mean(list(flow_on_ground_truth.values())) # average #flow-on across all AS
        n_flow_on_median = np.percentile(list(flow_on_ground_truth.values()), 50) # median #flow-on across all AS        
        n_peer = len([k for k in nodes.keys() if self.id[:2] in k]) # number of lb nodes
        chosen_n_flow_on = flow_on_ground_truth[chosen_id] # actual #flow-on on the chosen AS
        n_flow_on_obs_avg = chosen_n_flow_on / n_peer # theoretical average observed #flow-on on each LB node
        n_flow_on_obs = self._counters['n_flow_on'][chosen_id]
        flow.note.update({
            'is_shortest': chosen_n_flow_on == n_flow_on_min,
            'is_shorter_than_avg': chosen_n_flow_on < n_flow_on_avg,
            'is_shorter_than_median': chosen_n_flow_on < n_flow_on_median,
            'deviation': (n_flow_on_obs - n_flow_on_obs_avg)/(n_flow_on_obs_avg+1e-9),
        })
        if self.debug > 2: 
            print('-'*20)
            print(flow.note, "n_peer: {}".format(n_peer))
            print("chosen_n_flow_on={}".format(chosen_n_flow_on))
            print("n_flow_on_obs_avg={}".format(n_flow_on_obs_avg))
            print("n_flow_on_obs={}".format(n_flow_on_obs))
            print(self._counters['n_flow_on'])


    def receive(self, ts, flow, nodes):
        '''
        @brief:
            data plane implementation
        '''
        assert flow.nexthop == self.id
        flow.update_receive(ts, self.id)
        
        # random select 
        child_id, bucket_id = self.choose_child(flow, nodes, ts)

        # self.evaluate_decision_ground_truth(nodes, child_id, flow)
        if RENDER_RECEIVE: self.render_receive(ts, flow, child_id, nodes)

        # we hook reservoir sampling process with receive, whenever a new flow is received, we update the flow duration in reservoir sampling
        self.__update_res_fd(ts)

        if self._bucket_table_avail[bucket_id]: # bucket is available, register flow
            self._tracked_flows[flow.id] = (ts, bucket_id, child_id) # register t_receive and chosen AS id]
            self._counters['n_flow_on'][child_id] += 1

            self._bucket_table_avail[bucket_id] = False 
            if self.debug > 1: 
                print("bucket {} available, tracking flow {} -> node {}".format(bucket_id, flow.id, child_id))
                print('n_flow_on becomes', self._counters['n_flow_on'][self.child_ids])
        else:
            if self.debug > 1: print("bucket is not available, making flow untracked")
            self.n_untracked_flow += 1

        ts += self.get_process_delay() # add process delay
        flow.update_send(ts, '{}{}'.format(self.child_prefix, child_id)) # for now, we only implement for ecmp_random
        self.send(ts+self.get_t2neighbour(), flow)

        nodes['{}{}'.format(self.child_prefix, child_id)].update_pending_fct(flow)
        
    def expire_flow(self, ts, flow_id):
        '''
        @brief:
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

    def add_child(self, child_id, weights=[1]):
        if not isinstance(child_id, list): child_id = [child_id] # convert single as id to a list
        if not weights:
            weights = [np.mean(self.weights[self.child_ids])] * len(child_id)
        if not isinstance(weights, list): weights = [weights] # convert single weight to a list
        # sanity check
        assert len(child_id) == len(weights)
        assert len(set(child_id) - set(range(self.max_n_child))) == 0 and len(set(child_id) - set(self.child_ids)) == len(child_id) # child_id to add should be within legal AS range and beyond current active AS set
        self.child_ids += child_id
        for id_, w_ in zip(child_id, weights):
            self.weights[id_] = w_

        self.generate_bucket_table() # update bucket table

    def render_receive(self, ts, flow, chosen_child, nodes=None):
        self.render(ts, nodes)
        print('{:<30s}'.format('Chosen AS: ')+'{} for flow {}'.format(chosen_child, flow.id))

    def render(self, ts, nodes=None):
        '''
        @brief:
            print some observations and ground truths in a readable way
        '''
        obs = self.get_observation(ts)

        print('='*10, 'LB Render', '='*10)
        print('{:<30s}'.format('LB Node: ')+'{}'.format(self.id))
        print('{:<30s}'.format('Time: ')+'{:.6f}s'.format(time.time()-self.T0))
        print('{:<30s}'.format('Sim. Time: ')+'{:.6f}s'.format(ts))
        # print('{:<30s}'.format('{} ID:'.format(self.child_prefix))+' |'.join(
            # [' {:> 7d}'.format(i) for i in self.child_ids]))
        print('{:<30s}'.format('Current Weight:')+' |'.join(
            [' {:> 7.3f}'.format(self.weights[i]) for i in self.child_ids]))
        print('{:<30s}'.format('Tracked On Flow:')+' |'.join(
            [' {:> 7.0f}'.format(obs['n_flow_on'][i]) for i in self.child_ids]))
        print('{:<30s}'.format('Total untracked flows:')+str(self.n_untracked_flow))
        
        if nodes:
            print('{:<30s}'.format('Actual On Flow:')+' |'.join(
                [' {:> 7.0f}'.format(nodes['{}{}'.format(self.child_prefix, i)].get_n_flow_on()) for i in self.child_ids]))
            print('{:<30s}'.format('Number of Workers:')+' |'.join(
                [' {:> 7.0f}'.format(nodes['{}{}'.format(self.child_prefix, i)].n_worker) for i in self.child_ids]))

        # if self.debug > 1:
            # print(obs)
        # reward_field = np.array([obs[REWARD_FEATURE][i] for i in self.child_ids])
        # print('{:<30s}'.format('Reward Field:')+' |'.join(
        #     [' {:> 7.3f}'.format(v_) for v_ in reward_field]))
        # print('{:<30s}'.format('Reward: ')+'{:.6f}'.format(self.reward_fn(reward_field)))

    def step(self, ts, nodes=None):
        if RENDER: self.render(ts, nodes)
        t_delay = self.get_process_delay()
        # self.register_event(ts + t_delay, 'lb_update_bucket', {'node_id': self.id})
        self.register_event(ts + t_delay + self.lb_period, 'lb_step', {'node_id': self.id})

    def summarize(self):
        '''
        @brief:
            reduce flows' information during one episode
        '''

        res = {}

        res['n_untracked'] = self.n_untracked_flow

        return res
        
class NodeClient(Node):

    def __init__(
        self, 
        id, 
        child_ids, 
        app_config={},
        ip_prefix=None, 
        ip_mask=None,
        child_prefix='lb',
        debug=0
        ):
        '''
        @params:
            id: node id
            data_plane_buffer: all flows are pushed to dp_buffer
            control_plane_buffer: all events are pushed to cp_buffer
            nexthops: nexthop of the client flows are chosen randomly among these nexthops

        '''
        super(self.__class__, self).__init__(id, debug=debug)
        self._init_app_config = APPLICATION_CONFIG_TEMPLATE
        assert set(app_config.keys()) - set(["rate", "n_stage", "cpu_distribution", "io_distribution", "first_stage"]) == set()
        self._init_app_config.update(app_config)
        self.app_engine = Application(**self._init_app_config)
        if not isinstance(child_ids, list): child_ids = [child_ids]
        assert len(child_ids) > 0
        self.nexthops = ['{}{}'.format(child_prefix, i) for i in child_ids]
        self.ip_prefix = ip_prefix
        self.ip_mask = ip_mask
        self.summary = []  # organized by episode
        self.flows = []
        self.reset()

    def reset(self):
        self.app_config = self._init_app_config
        self.app_engine.update_config(**self.app_config)
        
        self.summarize()
        self.n_flow = 0
        del self.flows
        self.flows = []

        # kickoff by dispatching the first flow
        self.dispatch_flow(random.uniform(1e-3, 1e-1))

    def summarize(self):
        '''
        @brief:
            reduce flows' information during one episode
        '''
        if len(self.flows) == 0: 
            print("no flow registered")
            return

        if self.debug > 0:
            for flow in self.flows:
                print(flow.get_info())

        # iterate through all the flows
        flow_reject= []
        all_fct_expected, all_plt_expected, all_fct, all_plt = np.empty(
            0), np.empty(0), np.empty(0), np.empty(0)
        res, lr_pairs_dict = {}, {}
        # note_is_info = {
        #     'is_shortest': 0,
        #     'is_shorter_than_avg': 0,
        #     'is_shorter_than_median': 0,
        # }
        # note_deviation = np.empty(0)
        for flow in self.flows:
            if 'reject' in flow.nexthop: flow_reject.append(flow)
            all_fct_expected = np.append(all_fct_expected, flow.get_fct_expected())
            all_fct = np.append(all_fct, flow.get_fct())
            all_plt = np.append(all_plt, flow.get_plt())
            all_plt_expected = np.append(all_plt_expected, all_plt[-1] - (all_fct[-1] - all_fct_expected[-1]))
            as_node = flow.path[-1]
            if as_node in lr_pairs_dict.keys():
                lr_pairs_dict[as_node].append((flow.tss[-2], flow.tss[-1]))
            else:
                lr_pairs_dict[as_node] = [(flow.tss[-2], flow.tss[-1])]
            # for k in note_is_info.keys():
                # if flow.note[k]: note_is_info[k] += 1
            # note_deviation = np.append(note_deviation, flow.note['deviation'])
        
        # process gathered info
        n_flow = len(self.flows)
        res['n_reject'] = len(flow_reject)
        res['reject_ratio'] = res['n_reject'] / n_flow
        # for k, v in note_is_info.items():
        #     res['choice_{}_ratio'.format(k.lstrip('is_'))] = v / n_flow
        # res['n_flow_on_deviation_avg'] = note_deviation.mean()
        # res['n_flow_on_abs_deviation_avg'] = np.absolute(note_deviation).mean()
        res['all_fct'] = all_fct
        # res['all_plt'] = all_plt
        res.update(reduce_load(lr_pairs_dict))
        res.update(reduce_fct('fct', all_fct, all_fct_expected))
        res.update(reduce_fct('plt', all_plt, all_plt_expected))
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                res[k] = v.tolist()
        self.summary.append(res)
        return res
        
    def receive(self, ts, flow, nodes):
        assert self.id in flow.nexthop
        flow.update_receive(ts, self.id, terminal=True)
        self.flows.append(flow)
        for node_ in flow.path:
            if 'lb' in node_:
                # notice and expire the flow in all LB nodes
                self.register_event(ts+self.get_t2neighbour(), 'lb_expire_flow', {'node_id': node_, 'flow_id': flow.id})

    def generate_flow(self, ts):
        fields = []
        if self.ip_prefix:
            fields.append(generate_ip_random(self.ip_prefix, self.ip_mask))
            fields.append(generate_port_random())
        fct, first_stage, dt_next = self.app_engine.get_query()
        flow = Flow(
            '-'.join([self.id, str(self.n_flow)]),
            ts,
            random.choice(self.nexthops),
            fct,
            first_stage,
            fields
        )
        return flow, dt_next

    def dispatch_flow(self, ts):
        flow, dt_next = self.generate_flow(ts)
        self.n_flow += 1

        if self.debug > 1:
            print("|| @Client {}: dispatching flow {}".format(self.id, flow.get_info()))
        self.send(ts+self.get_process_delay(), flow)
        # register next flow generation event
        ts_next = ts + dt_next
        self.register_event(
            ts_next,
            'clt_send', 
            {'node_id': self.id})
        if self.debug > 0:
            print(">> ({:.3f}s) @client {} dispatch flow {}, next at {:.3f}s".format(ts, self.id, flow.get_info(), ts_next))

    def update_in_traffic(self, app_config):
        '''
        @brief:
            triggered by event 'clt_update_in_traffic'
        '''
        self.app_config.update(app_config)
        if self.debug > 0:
            print(">> @client {} update input traffic info to {}".format(
                self.id, self.app_config))
        self.app_engine.update(**self.app_config)  

# a global queue that stores all events
event_buffer = PriorityQueue()
