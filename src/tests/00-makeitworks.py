import subprocess
import numpy as np
import os
from multiprocessing import Value, Pool
import time
from pathlib import Path

n_thread_max = 2
counter = None
query_rate_list = [0.8]

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
            print(cmd_preamable)
            print('')
            cmd = cmd_preamable + \
                ' --lambda {0:.3f} -w {1} > {1}/test.log'.format(
                    rate, log_folder)
            final_task.append((cmd, log_folder))


    return final_task


seed = 45

methods = [
    "rlb-sac", # SAC model
]

n_lb = [1]
n_ass = [2]
setup_fmt = '{}lb-{}as-{}-hidden-{}-reward-{}-feature'
hidden_dims = [32, 64, 128, 512]
rewards = [0, 1, 2]
features = ['res_fct_avg', 'res_fd_avg', 'res_fd_avg_disc', 'res_fct_avg_disc']
n_flow_total = int(1000)

#--- other options ---#
# add ' --lb-bucket-size {}'.format(bucket_size) to change bucket size
# add ' --lb-period {}'.format(lb_period) to change bucket size


if __name__ == "__main__":  # confirms that the code is under main function
    tasks = []
    counter = Value('i', 0)
    T0 = time.time()

    experiment_name = 'first-impression-dump-all'
    root_dir = '../data/simulation/'
    data_dir = root_dir+experiment_name

    for n_lb in n_lb:
        for n_as in n_ass:
            for hidden_dim in hidden_dims:
                for reward in rewards:
                    for feature in features:
                        setup = setup_fmt.format(
                            n_lb, n_as, hidden_dim, reward, feature)
                        print(setup)
                        cmd_preamable = 'python3 run.py --n-lb {} --n-as {} --n-flow {}  --hidden-dim {} --reward-option {} --reward-feature {} --dump-all'.format(
                            n_lb, n_as, n_flow_total, hidden_dim, reward, feature)
                        for method in methods:
                            cmd = cmd_preamable + ' -m {}'.format(method)
                            log_folder = '/'.join([data_dir, setup, method])
                            tasks.append([cmd, log_folder])
                            Path(log_folder).mkdir(parents=True, exist_ok=True)
                            print('task : {}', cmd)
    final_tasks = add_rates(tasks, query_rate_list)

    final_tasks.append([('python3 run.py --n-lb 1 --n-as 6 --n-flow 20000 --dump-all -m ecmp --lambda 0.800 -w ../data/simulation/first-impression-dump-all/1lb-6as/ecmp/rate0.800 > ../data/simulation/first-impression-dump-all/1lb-6as/ecmp/rate0.800/test.log', '../data/simulation/first-impression-dump-all/1lb-6as/ecmp/rate0.800'), ('python3 run.py --n-lb 1 --n-as 6 --n-flow 20000 --dump-all -m wcmp --lambda 0.800 -w ../data/simulation/first-impression-dump-all/1lb-6as/wcmp/rate0.800 > ../data/simulation/first-impression-dump-all/1lb-6as/wcmp/rate0.800/test.log', '../data/simulation/first-impression-dump-all/1lb-6as/wcmp/rate0.800'), ('python3 run.py --n-lb 1 --n-as 6 --n-flow 20000 --dump-all -m lsq --lambda 0.800 -w ../data/simulation/first-impression-dump-all/1lb-6as/lsq/rate0.800 > ../data/simulation/first-impression-dump-all/1lb-6as/lsq/rate0.800/test.log', '../data/simulation/first-impression-dump-all/1lb-6as/lsq/rate0.800')])

    total_task = len(final_tasks)
    # for t in final_tasks:
    #     print(t)
    print('total tasks = {}'.format(total_task))
    pool_handler(tuple(final_tasks))
