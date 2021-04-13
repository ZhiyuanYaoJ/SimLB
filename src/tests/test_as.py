import sys
sys.path.insert(0, '..')
from common.entities import *
from random import expovariate
from numpy.random import exponential
from heapq import nsmallest
import operator

# initialization
flows = [] # a list of flows
src_node = 'clt'
poisson_lambda = 10
n_flow = 100
ts = 0
fct_mu = 0.5
as_id = 'as0'
dp_buffer = PriorityQueue()
n_worker = 2
multithread_level=2
node_as = NodeAS(as_id, dp_buffer, n_worker, multithread_level)
DEBUG = True
flow_client = []
change_capacity_tss = [6.099, 9.23, 1e6]
change_capacity_n_worker = [4, 3, None,]
change_capacity_mt_level = [1, 2, None,]
change_loc = 0

# generate a list of flows to test
for i in range(n_flow):
    ts += expovariate(poisson_lambda)
    fct = exponential(fct_mu)
    flow_ = Flow('-'.join([src_node, str(i)]), ts, fct, as_id)
    flow_.path.append('lb0')
    flow_.tss.append(ts + 1e-3)
    flow_.tss.append(ts + 2e-3)
    print(flow_.get_info())
    dp_buffer.put((ts + 5e-3, flow_))

print("=" * 20)
t_last = 0
while dp_buffer.qsize() > 0:
    ts_, flow_ = dp_buffer.pop()
    if ts_ > change_capacity_tss[change_loc]:
        dp_buffer.put((ts_, flow_))
        node_as.update_capacity(
            random.uniform(t_last, ts_), 
            change_capacity_n_worker[change_loc],
            change_capacity_mt_level[change_loc], 
        )
        change_loc += 1
        ts_, flow_ = dp_buffer.pop()
    if flow_.nexthop == as_id:
        node_as.receive(ts_, flow_)
        print('process_speed:', node_as.process_speed)
        print('worker_queue'.join(['-'*5] * 2))
        node_as.queues['worker'].print_queue()
        print('wait_queue'.join(['-'*5] * 2))
        node_as.queues['wait'].print_queue()
        print('data plane buffer'.join(['-'*5] * 2))
        dp_buffer.print_queue()
        print(' 3 smallest '.join(['-'*5] * 2))
        print(sorted(dp_buffer.queue)[:3])
    else:
        flow_client.append(flow_)
    t_last = ts_
    print("=" * 20)

for f in flow_client:
    f.print_info()