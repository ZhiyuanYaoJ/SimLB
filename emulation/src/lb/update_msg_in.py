import shm_proxy as sm
import numpy as np
import sys
import argparse
import time
import socket
import struct
from threading import Thread

#--- Arguments ---#

parser = argparse.ArgumentParser(description='Update msg_in to LB node.')

parser.add_argument('-m', action='store',
                    default='',
                    dest='method',
                    help='Method to update msg_in to LB node')

parser.add_argument('--list-weight', 
                    nargs='+', 
                    type=float,
                    default=[1.] * sm.N_AS,
                    dest='weights',
                    help='A list of weights')

parser.add_argument('-i', action='store',
                    default=0.5,
                    dest='interval',
                    help='Update interval')

parser.add_argument('-d', action='store_true',
                    default=False,
                    dest='dev',
                    help='Set dev mode and test offline without opening shared memory file')

parser.add_argument('--version', action='version',
                    version='%(prog)s 1.0')

#-- UTILS --#

def gen_alias(weights):
    '''
    @brief:
        generate alias from a list of weights (where every weight should be no less than 0)
    '''
    N = len(weights)
    avg = sum(weights)/N
    aliases = [(1, 0)]*N
    smalls = ((i, w/avg) for i,w in enumerate(weights) if w < avg)
    bigs = ((i, w/avg) for i,w in enumerate(weights) if w >= avg)
    small, big = next(smalls, None), next(bigs, None)
    while big and small:
        aliases[small[0]] = (float(small[1]), int(big[0]))
        big = (big[0], big[1] - (1-small[1]))
        if big[1] < 1:
            small = big
            big = next(bigs, None)
        else:
            small = next(smalls, None)
    return aliases

def __get_gt_worker(sid, asid, load, cpu, gt_sockets):
    '''
    @brief: 
        query ground truth info from servers
    @params:
        sid: sequence id of the frame
        asid: id of the AS to be processed
        ts: timestamp
    @dev:
        write parsed ground truth into a csv file
    '''        
    global shm_agent
    s = gt_sockets[asid]
    s.sendall(b'42\n') # send query to AS and wait for response
    data = s.recv(24) # 24 byte [cpu:memory:apache:as]
    gt = list(struct.unpack('dqii', data))
    cpu[shm_agent.as_map_hostid2vpp[asid]] = gt[0]
    load[shm_agent.as_map_hostid2vpp[asid]] = gt[2]

def get_gt(sid, _format, gt_sockets, n_step, cpu_load_register, queue_len_register):
    '''
    @brief:
        Create a bunch of threads each of which deals with one AS
    @note:  
        modify target function to parse in different ways
    '''
    global shm_agent
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    load = [0. ] * shm_agent.shm_n_bin
    cpu = [0. ] * shm_agent.shm_n_bin
    threads = [Thread(target=__get_gt_worker, args=(sid, asid, load, cpu, gt_sockets)) for asid in range(shm_agent.n_as)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    if _format == 'alias':
        non_zero_weights_id = [shm_agent.as_map_hostid2vpp[asid] for asid in range(shm_agent.n_as)[::-1]]
        # print("======")
        # print(">> bin:     {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d}".format(*non_zero_weights_id))
        cpu_load = np.array([cpu[_id] for _id in non_zero_weights_id])
        queue_len = np.array([load[_id] for _id in non_zero_weights_id])
        # print(">> #apache: {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d}".format(*non_zero_weights))
        cpu_load_register[:, sid%n_step] = cpu_load
        queue_len_register[:, sid%n_step] = queue_len

        if sid < n_step:
            weights = softmax(1/(queue_len+1))
        else:
            weights = softmax((queue_len_register/(cpu_load_register+1e-6)).mean(axis=1)/(queue_len+1))
        print(">> weights: {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f}".format(*weights))
        non_zero_alias = gen_alias(weights)
        # print(">> odds:    {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f}".format(*[_[0] for _ in non_zero_alias]))
        # print(">> alias:   {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d}".format(*[_[1] for _ in non_zero_alias]))
        final_alias = [(1., 0)] * shm_agent.shm_n_bin
        for _alias, asid in zip(non_zero_alias, non_zero_weights_id):
            final_alias[asid] = (_alias[0], _alias[1])
        shm_agent.register_as_alias(sid, final_alias)
    else:
        shm_agent.register_as_score(sid, load)

#--- Methods ---#

def static_ws(weights):
    '''
    @brief:
        update weights in the Alias Method form for once
    '''
    global shm_agent
    alias = [(1., 0)] * shm_agent.shm_n_bin
    non_zero_weights = [v for v in weights if v > 0]
    non_zero_weights_id = [i for i, v in enumerate(weights) if v > 0]
    
    non_zero_alias = gen_alias(non_zero_weights)
    print("======")
    print(">> bin:     {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d}".format(*non_zero_weights_id))
    print(">> weights: {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f}".format(*non_zero_weights))
    print(">> odds:    {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f} {:<0.3f}".format(*[_[0] for _ in non_zero_alias]))
    print(">> alias:   {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d}".format(*[_[1] for _ in non_zero_alias]))
            
    for _alias, asid in zip(non_zero_alias, non_zero_weights_id):
        alias[asid] = (_alias[0], _alias[1])
    # print(alias)
    shm_agent.register_as_alias(1, alias)

def active_probe(interval, _format='score'):
    gt_sockets = sm.get_sockets()
    sid = 0
    n_active_as = sm.GLOBAL_CONF['meta']['n_as']
    n_step = 8
    cpu_load_register = np.zeros((n_active_as, n_step))
    queue_len_register = np.zeros((n_active_as, n_step))
    while True:
        _t0 = time.time()
        get_gt(sid, _format, gt_sockets, n_step, cpu_load_register, queue_len_register)
        sid += 1
        dt = time.time() - _t0
        time.sleep(max(interval - dt, 0.))

#-- DEFINE MACRO & PARAMS --#


#-- main --#
if __name__ == '__main__':

    args = parser.parse_args()

    method_fc_mapper = {
        'static-ws': static_ws,
        'active-wcmp': active_probe,
        'active-po2': active_probe,
    }

    # make sure the method exists
    assert args.method in method_fc_mapper.keys()

    if 'ws' in args.method or 'wcmp' in args.method:
        args.format = 'alias'
    else:
        args.format = 'score'

    # initialize shared memory manager
    shm_agent = sm.Shm_Manager(sm.CONF_FILE)

    # initialize sequential number
    seq_id = 0

    method_kwargs_mapper = {
        'static-ws': {
            'weights': args.weights,
        },
        'active-wcmp': {
            'interval': args.interval,
            '_format': args.format,
        },
        'active-po2': {
            'interval': args.interval,
            '_format': args.format,
        },
    }

    # run method function and pass corresponding arguments
    method_fc_mapper[args.method](
        **method_kwargs_mapper[args.method]
        )