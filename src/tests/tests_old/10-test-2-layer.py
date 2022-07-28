import subprocess
import numpy as np
import os
from multiprocessing import Value, Pool
import time
from pathlib import Path

n_thread_max = 2
counter = None
query_rate_list = [0.7, 0.8, 0.9]

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


seed = 2

n_episode = 20
first_episode_id = 0
t_episode = 60
t_episode_inc = 5

if __name__ == "__main__":  # confirms that the code is under main function
    tasks = []
    counter = Value('i', 0)
    T0 = time.time()
    experiment_name = 'test-2-layer'
    root_dir = '../data/simulation/'
    data_dir = root_dir+experiment_name
    
    methods = [
        ["rlb-sac", 'lsq', False],
    ]
    configs = [
        (1,8,64),
    ]
    setup_fmt = '{}lbp-{}lbs-{}as'

    for config in configs:
        n_lbp, n_lbs, n_as = config
        setup = setup_fmt.format(n_lbp, n_lbs, n_as)
        print(setup)
        cmd_preamable = 'python3 run_hierarchical.py --n-lbp {} --n-lbs {} --n-as {} --max-n-child {} -t {} --t-inc {} --n-episode {} --dump-all'.format(
            n_lbp, n_lbs, n_as, n_as, t_episode, t_episode_inc, n_episode)
        for method in methods:
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
            
    methods = [
        'rlb-sac',
        # 'rlb-sac2',
        # 'rlb-sac-small',
        # 'rlb-sac-tiny2',
        'rlb-sac-tiny',
        'lsq',
        'sed',
        'wcmp',
        'ecmp',
        'srt',
    ]
    configs = [
        (1,64)
    ]
    setup_fmt = '{}lb-{}as'

    for config in configs:
        n_lb, n_as = config
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
            
    final_tasks = add_rates(tasks, query_rate_list)
    total_task = len(final_tasks)
    print('total tasks = {}'.format(total_task))
    pool_handler(tuple(final_tasks))
