import subprocess
import numpy as np
import os
from multiprocessing import Value, Pool
import time
from pathlib import Path

n_thread_max = 48
counter = None
# query_rate_list = np.array([0.115 * i for i in range(1, 6)] + [0.115 * 5 + 0.035 * i for i in range(
#     1, 5)] + [0.115 * 5 + 0.03 * 5 + 0.02 * i for i in range(1, 14)] + [1])[6::4]
query_rate_list = [0.8, 1]

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


seed = 46

methods = [
    # [method 1, method2, auto_clustering]
    #["rlb-sac", 'rlb-sac', False],
    ["wcmp", 'lsq', False],
]

# grid search dimensions
n_lbps = [1]
n_lbss = [2]
n_ass = [6]
n_episode = 1
setup_fmt = '{}lbp-{}lbs-{}as'
first_episode_id = 0
n_flow_total = int(1e4)
#--- other options ---#
# add ' --lb-bucket-size {}'.format(bucket_size) to change bucket size
# add ' --lb-period {}'.format(lb_period) to change load banlancer period


if __name__ == "__main__":  # confirms that the code is under main function
    tasks = []
    counter = Value('i', 0)
    T0 = time.time()

    experiment_name = 'first-impression-dump-all'
    root_dir = '../data/simulation/'
    data_dir = root_dir+experiment_name
    for n_lbp in n_lbps:
        for n_lbs in n_lbss:
            for n_as in n_ass:
                    setup = setup_fmt.format(n_lbp, n_lbs, n_as)
                    print(setup)
                    cmd_preamable = 'python3 run_hierarchical.py --n-lbp {} --n-lbs {} --n-as {} --n-flow {} --n-episode {} --first-episode-id {} --dump-all'.format(
                        n_lbp, n_lbs, n_as, n_flow_total, n_episode, first_episode_id)
                    for method in methods:
                        method1 = method[0]
                        method2 = method[1]
                        auto_clustering=method[2]
                        cmd = cmd_preamable + ' -m1 {}'.format(method1)
                        cmd = cmd + ' -m2 {}'.format(method2)
                        cmd = cmd + ' --auto-clustering'
                        #cmd = cmd + ' --user-conf {}'.format(user_conf)
                        log_folder = '/'.join([data_dir, setup, method1 + method2])
                        tasks.append([cmd, log_folder])
                        Path(log_folder).mkdir(parents=True, exist_ok=True)
                        print('task : {}', cmd)
    final_tasks = add_rates(tasks, query_rate_list)

    total_task = len(final_tasks)
    # for t in final_tasks:
    #     print(t)
    print('total tasks = {}'.format(total_task))
    pool_handler(tuple(final_tasks))
