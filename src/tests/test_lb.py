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
lb_id = 'lb0'
dp_buffer = PriorityQueue()
cp_buffer = PriorityQueue()
active_as = [0, 1, 2, 3, 5]
weights = {
    i: 1. for i in active_as
}
node_lb = NodeLB(lb_id, dp_buffer, cp_buffer, active_as)
unique, counts = np.unique(node_lb._bucket_table, return_counts=True)
n_i = dict(zip(unique, counts))
for i, n in n_i.items():
    print("as {}: {} ({:.2%})".format(i, n, n/node_lb.bucket_size))
DEBUG = 2
flow_as = []

# generate a list of flows to test and prepare data plane
for i in range(n_flow):
    ts += expovariate(poisson_lambda)
    fct = exponential(fct_mu)
    flow_ = Flow('-'.join([src_node, str(i)]), ts, fct, lb_id)
    print(flow_.get_info())
    dp_buffer.put((ts + 5e-3, flow_))

# prepare control plane
cp_buffer.put(Event(3., 'add_server', {'servers': [17, 18], 'weights': [0.5, 0.2]}), checkfull=False)
cp_buffer.put(Event(6., 'remove_server', {'servers': [
              17, 18], 'weights': [0.5, 0.2]}), checkfull=False)
cp_buffer.put(Event(float('inf'), 'end_of_the_world', {}), checkfull=False)
cp_buffer.put(Event(1e-6, 'lb_step', {}), checkfull=False)

# get first events from data/control plane
dp_next = dp_buffer.pop()
cp_next = cp_buffer.pop()

print("=" * 20)
t_last = 0
while dp_buffer.qsize() > 0 and cp_buffer.qsize() > 1:
    if dp_next[0] < cp_next.time:
        ts_, flow_ = dp_next
        dp_next = dp_buffer.pop()

        if flow_.nexthop == lb_id:
            node_lb.receive(ts_, flow_)
            print('data plane buffer'.join(['-'*5] * 2))
            dp_buffer.print_queue()
            print(' 3 smallest '.join(['-'*5] * 2))
            for element in sorted(dp_buffer.queue)[:3]:
                print('>> ({:.3f}s)'.format(element[0]), element[1].get_info())
        else:
            flow_as.append(flow_)
            t_end = ts_+flow_.fct
            if t_end < cp_next.time:
                cp_buffer.put(cp_next)
                cp_next = Event(t_end, 'expire_flow', {'fid': flow_.id})
            else:
                cp_buffer.put(Event(t_end, 'expire_flow', {'fid': flow_.id}))
        t_last = ts_
    else:
        ts_, event, kwargs = cp_next
        cp_next = cp_buffer.pop()
        if event == 'add_server':
            node_lb.add_as(kwargs['servers'], kwargs['weights'])
        elif event == 'remove_server':
            node_lb.remove_as(kwargs['servers'])
        elif event == 'lb_step':
            node_lb.step(ts_)
            continue
        elif event == 'expire_flow':
            node_lb.expire_flow(ts_, kwargs['fid'])
        else:
            raise NotImplementedError
        print(">> Event {} <<".format(event))
        # node_lb.render(ts_)
    print("=" * 20)

for f in flow_as:
    f.print_info()
