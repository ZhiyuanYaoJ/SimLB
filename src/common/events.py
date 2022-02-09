# ---------------------------------------------------------------------------- #
#                                  Description                                 #
# This file defines events and their corresponding reactions, each event's     #
# name is the name of a function                                               #
#   - Add/remove ASs                                                           #
#   - Change AS capacity                                                       #
#   - Update bucket table                                                      #
#   - Step (Render and generate new weights)                                   #
# ---------------------------------------------------------------------------- #

from config.global_conf import *
from common.entities import Event, NodeAS, event_buffer

def dp_receive(nodes, ts, flow):
    nexthop = flow.nexthop.split('-')[0]
    if DEBUG > 0:
        print(">> ({:.3f}s) @node {} receives {}".format(ts, nexthop, flow.get_info()))
    nodes[nexthop].receive(ts, flow, nodes)

def as_update_capacity(nodes, ts, node_ids, n_worker, mp_level):
    for node_id in node_ids:
        nodes[node_id].update_capacity(ts, n_worker, mp_level)

def as_try_remove(nodes, ts, node_id):
    global event_buffer
    t_end = nodes[node_id].queues['worker'].peek_n(1, reverse=True)
    if len(t_end) == 0:
        del nodes[node_id]
        if DEBUG > 1:
            print("remove node", node_id)
    else:
        t_end = t_end[0][0]
        event_buffer.put(Event(t_end+1e-6, 'as_try_remove', {'node_id': node_id})) 
        if DEBUG > 1:
            print("there are still {} pending queries @node {}, try next time at {:.3f}s".format(nodes[node_id].queues['worker'].qsize()+nodes[node_id].queues['wait'].qsize(), node_id, t_end))

def as_periodic_log(nodes, ts, node_ids, interval):
    global event_buffer
    print('Periodic check: '+'|'.join(['{} {:.6f}'.format(node_id, nodes[node_id].get_t_rest_total(ts)) for node_id in node_ids]))
    event_buffer.put(Event(ts+interval, 'as_periodic_log', 'sys-admin', {'node_ids': node_ids, 'interval': interval}))

def lb_update_bucket(nodes, ts, node_id):
    '''
    @brief:
        triggered by 'lb_update_bucket'
    '''
    nodes[node_id].generate_bucket_table()

def lb_step(nodes, ts, node_id):
    nodes[node_id].step(ts, nodes)

def lb_expire_flow(nodes, ts, node_id, flow_id):
    if DEBUG > 0:
        print(">> ({:.3f}s) @node {} expire flow {}".format(ts, node_id, flow_id))
    nodes[node_id].expire_flow(ts, flow_id)


def lb_add_server(nodes, ts, lbs, ass, weights, n_workers=N_WORKER_BASELINE, mp_levels=1, max_client=AS_MAX_CLIENT):
    '''
    @brief:
        a control plane event, when we want to add AS nodes and associate them to some LBs. 
    '''
    if isinstance(n_workers, int):
        n_workers = [n_workers] * len(ass)
    if isinstance(mp_levels, int):
        mp_levels = [mp_levels] * len(ass)
    if isinstance(max_client, int):
        max_clients = [max_client] * len(ass)
    for as_id, n_worker, mp_level, max_client in zip(ass, n_workers, mp_levels, max_clients):
        as_id = 'as{}'.format(as_id)
        nodes[as_id] = NodeAS(as_id, n_worker, mp_level, max_client)
    for lb in lbs:
        nodes['lb{}'.format(lb)].add_child(ass, weights)

def lb_remove_server(nodes, ts, lbs, ass):
    for lb in lbs:
        nodes['lb{}'.format(lb)].remove_child(ass)
    for as_id in ass:
        as_id = 'as{}'.format(as_id)
        ts += 1e-6
        event_buffer.put(Event(ts, 'as_try_remove',
                               {'node_id': as_id}), checkfull=False)

def clt_update_in_traffic(nodes, ts, node_id, app_config_new):
    if DEBUG > 0:
        print(">> ({:.3f}s) @node {} update in traffic".format(
            ts, node_id))
    nodes[node_id].update_in_traffic(app_config_new)

def clt_dispatch(nodes, ts, node_id):
    assert 'clt' in node_id
    nodes[node_id].dispatch_flow(ts)

def clt_send(nodes, ts, node_id):
    if DEBUG > 0:
        print(">> ({:.3f}s) @node {} dispatch flow".format(
            ts, node_id))
    nodes[node_id].dispatch_flow(ts)
