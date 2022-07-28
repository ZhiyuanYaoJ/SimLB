import subprocess
import numpy as np
import os
from multiprocessing import Value, Pool
import time
from pathlib import Path

n_thread_max = 2
counter = None
query_rate_list = [0.9]

def init(args):
    ''' store the counter for later use '''
    global counter
    counter = args

def create_path(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)


def subprocess_cmd(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    return proc_stdout.decode("utf-8")


def work_log(work_data):
    global counter
    cmd, filename = work_data
    t0 = time.time()
    print("[{:6.3f}s]  Process {}...".format(t0-T0, filename))
    subprocess_cmd(cmd)

    # += operation is not atomic, so we need to get a lock:
    with counter.get_lock():
        counter.value += 1
    task_num = counter.value
    percentage = task_num/total_task
    print("[{:6.3f}s]  Process {:>50s} finished (duration: {:>6.3f}s | {:>7d}/{:<7d} - {:>5.2%})".format(
        t0-T0, filename, time.time()-t0, task_num, total_task, percentage))


def pool_handler(work):
    p = Pool(n_thread_max, initializer=init, initargs=(counter, ))
    p.map(work_log, work)


def add_rates(tasks, rates):
    '''
    @params:
        work_data: a tuple consists of (cmd header, and father folder name)
    '''

    final_task = []

    for cmd_preamable, foldername in tasks:
        for rate in rates:
            log_folder = os.path.join(foldername, 'rate{:.3f}'.format(rate))
            Path(log_folder).mkdir(parents=False, exist_ok=True)
            cmd = cmd_preamable + \
               ' --lambda {0:.3f} -w {1} > {1}/test.log'.format(
                    rate, log_folder)

            final_task.append((cmd, log_folder))

    return final_task


seed = 89

n_episode = 15
first_episode_id = 0
t_episode = 60
t_episode_inc = 5

if __name__ == "__main__":  # confirms that the code is under main function
    tasks = []
    counter = Value('i', 0)
    T0 = time.time()
    experiment_name = 'Algorithms_comparison'
    root_dir = '../data/simulation/'
    data_dir = root_dir+experiment_name
    
    methods = [
        #=== rule ===#
        # "ecmp", # Equal-Cost Multi-Path (ECMP)
        # "wcmp", # Weighted-Cost Multi-Path (WCMP)
        # "lsq", # Local shortest queue (LSQ)
        # "lsq2", # LSQ + power-of-2-choices
        # "sed", # Shortest Expected Delay
        # "sed2", # LSQ + power-of-2-choices
        # "srt", # Shortest Remaining Time (SRT) (Layer-7)
        # "srt2", # SRT + power-of-2-choices
        #"gsq", # Global shortest queue (GSQ) (Layer-7)
        #"gsq2", # GSQ + power-of-2-choices·
        # "active-wcmp", # Spotlight, adjust weights based on periodic polling
        #=== heuristic ===#
        # "aquarius", # Aquarius, 
        # "hlb", # Hybrid LB (HLB), Aquarius replacing alpha by Kalman filter
        # "hlb2", # HLB + power-of-2-choices
        # "hlb-ada", # HLB + adaptive sensor error
        # "hermes", #hermes
        # "rs", # reservoir sampling #flow
        # "rs2", # reservoir sampling #flow + power-of-2
        # "geom", # geometry-based algorithm
        #"geom-w", # geometry-based algorithm
        # "prob-flow", # geometry-based algorithm
        #"prob-flow-w", # geometry-based algorithm
        # "prob-flow2", # geometry-based algorithm
        #"prob-flow-w2", # geometry-based algorithm
        #"geom-sed", # geometry-based algorithm
        #"geom-sed-w", # geometry-based algorithm
        # === reinforcement learning ===#
        # "rlb-sac", # SAC model
        # "rlb-sac-tiny", # SAC tiny-model
        'rlb-sac2'
    ]
    methods_hierarchical = [
        # === Hierarchicak methods learning ===#
        # ("rlb-sac","lsq", False), # Top LB method, Secondary LB method, clustering agent?
    ]
    configs = [ 
        # (1,64), # 1LB, 64 servers
        (1,16), # 1LB, 16 servers
        # (1,4) # 1LB, 4 servers
    ]
    configs_hierarchical = [
        (1,8,64), #1 primary LB, 8 secondary LBs, 64 servers
        # (1,4,16), #1 primary LB, 4 secondary LBs, 16 servers
        # (1,2,4), #1 primary LB, 2 secondary LBs, 4 servers
    ]
    
    for config in configs:
        n_lb, n_as = config
        setup_fmt = '{}lb-{}as'
        setup = setup_fmt.format(n_lb, n_as)
        print(setup)
        cmd_preamable = 'python3 run.py --n-lb {} --n-as {} --max-n-child {} -t {} --t-inc {} --n-episode {} --dump-all'.format(
            n_lb, n_as, n_as, t_episode, t_episode_inc, n_episode)
        for method in methods:
            cmd = cmd_preamable + ' -m {}'.format(method)
            log_folder = '/'.join([data_dir, setup, method])
            tasks.append([cmd, log_folder])
            Path(log_folder).mkdir(parents=True, exist_ok=True)
            print('task : {}', cmd)
            
    for config in configs_hierarchical:
        n_lbp, n_lbs, n_as = config
        setup_fmt = '{}lbp-{}lbs-{}as'
        setup = setup_fmt.format(n_lbp, n_lbs, n_as)
        print(setup)
        cmd_preamable = 'python3 run_hierarchical.py --n-lbp {} --n-lbs {} --n-as {} --max-n-child {} -t {} --t-inc {} --n-episode {} --dump-all'.format(
            n_lbp, n_lbs, n_as, n_as, t_episode, t_episode_inc, n_episode)
        for method in methods_hierarchical:
            method1 = method[0]
            method2 = method[1]
            cmd = cmd_preamable + ' -m1 {}'.format(method1)
            cmd = cmd + ' -m2 {}'.format(method2)
            if method[2] == True:
                cmd = cmd + ' --auto-clustering'
            log_folder = '/'.join([data_dir, setup, method1 + method2])
            tasks.append([cmd, log_folder])
            Path(log_folder).mkdir(parents=True, exist_ok=True)
            print('task : {}', cmd)
     
    final_tasks = add_rates(tasks, query_rate_list)
    total_task = len(final_tasks)
    print('total tasks = {}'.format(total_task))
    pool_handler(tuple(final_tasks))
