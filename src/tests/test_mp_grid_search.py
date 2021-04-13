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

n_thread_max = 36
counter = None
query_rate_list = np.array([0.115 * i for i in range(1, 6)] + [0.115 * 5 + 0.035 * i for i in range(
    1, 5)] + [0.115 * 5 + 0.03 * 5 + 0.02 * i for i in range(1, 14)] + [1])[::2]


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


def read_file(filename):
    lines = []
    with open(filename, 'r') as myfile:
        lines = [line.rstrip('\n') for line in myfile]
    return lines


def get_n_reject_avg(folder_name):
    '''
    @brief:
        gather average number of rejected flows across a set of episodes in a given setup
    '''
    lines = read_file(os.path.join(folder_name, 'test.log'))
    n_reject = [int(l.split(' ')[-1])
                for l in lines if 'number of rejected flow in total:' in l]
    return np.mean(n_reject)


def work_log(work_data):
    '''
    @params:
        work_data: a tuple consists of (cmd header, and father folder name)
    '''
    global counter, query_rate_list, T0, total_task, max_lambda_rate

    cmd_preamable, foldername = work_data

    # we first run through pre-defined list of traffic rates
    for rate in query_rate_list:
        log_folder = os.path.join(foldername, 'rate{:.3f}'.format(rate))
        Path(log_folder).mkdir(parents=False, exist_ok=True)
        cmd = cmd_preamable + \
            ' --lambda {0:.3f} -w {1} > {1}/test.log'.format(rate, log_folder)
        t0 = time.time()
        print("[{:6.3f}s]  Process {}...".format(t0-T0, log_folder))
        subprocess_cmd(cmd)

        # += operation is not atomic, so we need to get a lock:
        with counter.get_lock():
            counter.value += 1
        task_num = counter.value
        percentage = task_num/total_task
        print("[{:6.3f}s]  Process {:>50s} finished (duration: {:>6.3f}s | {:>7d}/{:<7d} - {:>5.2%})".format(
            t0-T0, log_folder, time.time()-t0, task_num, total_task, percentage))

    n_reject = get_n_reject_avg(log_folder)

    # then we run until we have rejected flows
    while n_reject == 0 and rate <= max_lambda_rate:
        rate += 0.05
        log_folder = os.path.join(foldername, 'rate{:.3f}'.format(rate))
        Path(log_folder).mkdir(parents=False, exist_ok=True)
        cmd = cmd_preamable + \
            ' --lambda {0:.3f} -w {1} > {1}/test.log'.format(rate, log_folder)
        t0 = time.time()
        print("[{:6.3f}s]  Process {}...".format(t0-T0, log_folder))
        subprocess_cmd(cmd)

        n_reject = get_n_reject_avg(log_folder)
        t1 = time.time()
        print("[{:6.3f}s]  Process {:>50s} finished (duration: {:>6.3f}s | w/ n_rejected_flow {:.2f})".format(
            t1-T0, log_folder, t1-t0, n_reject))

    print("[{:6.3f}s] >> Batch process for {:>50s} finished (highest traffic rate {:.2f}) <<".format(
        time.time()-T0, foldername, rate))


def pool_handler(work):
    p = Pool(n_thread_max, initializer=init, initargs=(counter, ))
    p.map(work_log, work)


seed = 42

methods = [
    #'ecmp',  # ECMP
    # 'weight',  # Static Weight
    'lsq',  # Local shortest queue
    # 'lsq2',  # Local shortest queue + power-of-2-choices
    # 'heuristic',  # Heuristic
    # 'kf1d',  # 1D Kalman-Filter-Based LB
    # 'weightlsq',  # LSQ
    # 'weightlsq2',  # LSQ + power-of-2-choices
    # 'heuristiclsq',  # LSQ
    # 'heuristiclsq2',  # LSQ + power-of-2-choices
    # 'kf1dlsq',  # LSQ
    # 'kf1dlsq2',  # LSQ + power-of-2-choices
    'weightlsq-dev',  # LSQ
    #    'weightlsq2-dev',  # LSQ + power-of-2-choices
    # 'heuristiclsq-dev',  # LSQ
    # 'heuristiclsq2-dev',  # LSQ + power-of-2-choices
    'kf1dlsq-dev',  # LSQ
    #    'kf1dlsq2-dev',  # LSQ + power-of-2-choices
    'oracle',  # 
]

# grid search dimensions
n_lbs = [4, 12]
n_ass = [32, 128]
n_workers = [1]
# n_workers = [1, 4]
b_offsets = [1]
# b_offsets = [1, 2]
fct_mu = 0.75
# fct_mus = [0.25, 0.5, 0.75, 1.0]
setup_fmt = '{}lb-{}as-{}worker-{}stage-exp-{:.2f}cpumu'
kf_sensor_stds = [0.4, 0.6, 0.8]
n_process_stage = 3
n_episode = 5
max_lambda_rate = 1.1
fct_io = 0.75  # fixed average FCT of IO process
n_flow_total = int(5e4)
T0 = time.time()

if __name__ == "__main__":  # confirms that the code is under main function
    tasks = []
    counter = Value('i', 0)

    experiment_name = 'test-low-rate'
    root_dir = '../../data/simulation/'
    data_dir = root_dir+experiment_name

    for n_lb in n_lbs:
        for n_as in n_ass:
            for n_worker in n_workers:
                setup = setup_fmt.format(
                    n_lb, n_as, n_worker, n_process_stage, fct_mu)
                if n_process_stage > 1:
                    setup += '-{:.2f}iomu'.format(fct_io)
                cmd_preamable = 'python3 run.py --n-lb {} --n-as {} --n-worker {} --cpu-fct-mu {} --process-n-stage {} --io-fct-mu {} --n-flow {}'.format(
                    n_lb, n_as, n_worker, fct_mu, n_process_stage, fct_io, n_flow_total)
                for method in methods:
                    if 'kf1d' in method:
                        for b_offset in b_offsets:
                            for kf_sensor_std in kf_sensor_stds:
                                method_alias = method + \
                                    '-std{:.1f}-b{}'.format(kf_sensor_std,
                                                            b_offset)
                                log_folder = '/'.join([data_dir,
                                                       setup, method_alias])
                                cmd = cmd_preamable + \
                                    ' -m {} --kf-sensor-std {} --b-offset {}'.format(
                                        method, kf_sensor_std, b_offset)
                                tasks.append([cmd, log_folder])
                                Path(log_folder).mkdir(
                                    parents=True, exist_ok=True)
                    elif 'weightlsq' in method:
                        for b_offset in b_offsets:
                            method_alias = method + '-b{}'.format(b_offset)
                            log_folder = '/'.join([data_dir,
                                                   setup, method_alias])
                            cmd = cmd_preamable + \
                                ' -m {} --b-offset {}'.format(method, b_offset)
                            tasks.append([cmd, log_folder])
                            Path(log_folder).mkdir(parents=True, exist_ok=True)
                    else:
                        log_folder = '/'.join([data_dir, setup, method])
                        cmd = cmd_preamable + ' -m {}'.format(method)
                        tasks.append([cmd, log_folder])
                        Path(log_folder).mkdir(parents=True, exist_ok=True)

    total_setup = len(tasks)
    total_task = total_setup * len(query_rate_list)
    # for t in tasks:
    #     print(t)
    print('total setup = {}'.format(total_setup))
    print('estimated tasks = {}'.format(total_task))
    pool_handler(tuple(tasks))
