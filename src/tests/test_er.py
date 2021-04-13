# ---------------------------------------------------------------------------- #
#                                  Description                                 #
# This file aims at testing client node functionalities, including following   #
# aspects:                                                                     #
#   - dispatching new flows until the world ends                               #
#   - receive flows and register in the received flow list                     #
#   - expire flows in LB nodes which store states in bucket table              #
#   - summary fct info after one episode                                       #
#   - update in_traffic_info                                                   #
# ---------------------------------------------------------------------------- #


import sys
sys.path.insert(0, '..')
from common.entities import *
from common.events import *

# initialization
in_traffic_info = {
    'rate': 10,
    'type': 'exp',
    'mu': 0.6
}
t_episode = 9.
ts_ = 0
er_id = 'er0'
lb_ids = [0, 1]
clt_id = 'clt0'
active_as = [0, 1, 2, 3, 5]
weights = {
    i: 1. for i in active_as
}
node_clt = NodeClient(clt_id, [0], in_traffic_info=in_traffic_info, child_prefix='er', debug=1)
node_er = NodeStatelessLB(er_id, lb_ids, child_prefix='lb', debug=2)
nodes = {
    clt_id: node_clt,
    er_id: node_er,
}
for lb_id in lb_ids:
    lb_id = 'lb{}'.format(lb_id)
    nodes[lb_id] = NodeLB(lb_id, active_as)
for as_id in active_as:
    as_id = 'as{}'.format(as_id)
    nodes[as_id] = NodeAS(as_id, )

# prepare control plane to check add and remove
event_buffer.put(Event(3., 'lb_add_server', {'lbs': [0], 'ass': [17, 18], 'weights': [
                 0.5, 0.2], 'n_workers': N_WORKER_BASELINE, 'mp_levels': 1, 'wait_queue_lens': AS_WAIT_QUEUE_LEN}), checkfull=False)
event_buffer.put(Event(6., 'lb_remove_server', {'lbs': [0], 'ass': [
              17, 18]}), checkfull=False)
event_buffer.put(Event(3.2, 'clt_update_in_traffic', {
                 'node_id': clt_id, 'in_traffic_info_new': {'rate': 20, 'type': 'normal', 'mu': 1.0, 'std': 0.3}}))

# get first events from data/control plane
e_next = event_buffer.pop()

print("=" * 20)
while ts_ < t_episode:

    ts_, event, kwargs = e_next
    print("="*20, "Registered Events", "="*20)
    event_buffer.print_queue()
    print(">> Event {} <<".format(event))
    arg_keys = getfullargspec(eval(event))[0][2:]
    eval(event)(nodes, ts_, *[kwargs[k] for k in arg_keys])
    # node_lb.render(ts_)
    print("=" * 20)
    e_next = event_buffer.pop()


node_clt.summarize()
print(node_clt.summary)
