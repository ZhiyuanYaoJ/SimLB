# ---------------------------------------------------------------------------- #
#                                  Description                                 #
# This file runs w/ each thread one mechanism over a list of traffic rates,    #
# until there are rejected flows.                                              #
# ---------------------------------------------------------------------------- #

import subprocess
import numpy as np
import os
from multiprocessing import Value, Pool
import time
from pathlib import Path

n_thread_max = 6
counter = None
query_rate_list = [0.7, 0.9]


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


seed = 42
methods = [
    'ecmp',  # ECMP
    'weight',  # Static Weight
    'lsq',  # Local shortest queue (LSQ)
    'sed',  # LSQ
    'oracle',  # a god-like LB that knows remaining
    'gsq2',  # a god-like LB that knows remaining
    'hlb-ada',  # KF1d + LSQ w/ adaptive sensor error
    'active-wcmp',  # Active WCMP
]

# grid search dimensions
n_lbs = [4]
n_ass = [128]
n_workers = [1]
fct_pairs = [(0.5, 0.5), (0.75, 0.25), (0.25, 0.75)]  # fct_cpu & fct_io pairs
setup_fmt = '{}lb-{}as-{}worker-{}stage-exp-{:.2f}cpumu'
kf_sensor_stds = [0.4]
n_process_stage = 3
n_episode = 5
max_lambda_rate = 1.1
first_episode_id = 0
n_flow_total = int(8e4)
T0 = time.time()

if __name__ == "__main__":  # confirms that the code is under main function
    tasks = []
    counter = Value('i', 0)

    experiment_name = 'multi-stage-reduce'
    root_dir = '../data/simulation/'
    data_dir = root_dir+experiment_name

    for n_lb in n_lbs:
        for n_as in n_ass:
            for n_worker in n_workers:
                for fct_mu, fct_io in fct_pairs:
                    setup = setup_fmt.format(
                        n_lb, n_as, n_worker, n_process_stage, fct_mu)
                    if n_process_stage > 1:
                        setup += '-{:.2f}iomu'.format(fct_io)
                    print(setup)
                    cmd_preamable = 'python3 run.py --n-flow {} --n-lb {} --n-as {} --n-worker {} --cpu-fct-mu {} --process-n-stage {} --io-fct-mu {} --n-episode {} --first-episode-id {}'.format(
                        n_flow_total, n_lb, n_as, n_worker, fct_mu, n_process_stage, fct_io, n_episode, first_episode_id)
                    for method in methods:
                        log_folder = '/'.join([data_dir, setup, method])
                        cmd = cmd_preamable + ' -m {}'.format(method)
                        tasks.append([cmd, log_folder])
                        Path(log_folder).mkdir(parents=True, exist_ok=True)
    final_tasks = add_rates(tasks, query_rate_list)

    total_task = len(final_tasks)
    # for t in final_tasks:
    #     print(t)
    print('total tasks = {}'.format(total_task))
    pool_handler(tuple(final_tasks))
