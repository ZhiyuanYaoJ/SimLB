import subprocess
import numpy as np
import os
from multiprocessing import Value, Pool
import time


n_thread_max = 46
counter = None


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


t_episode = 200
seed = 42
n_as = 32
n_worker = 1
n_lb = 14

query_rate_list = np.array([0.115 * i for i in range(1, 6)] + [0.115 * 5 + 0.035 * i for i in range(
    1, 5)] + [0.115 * 5 + 0.03 * 5 + 0.02 * i for i in range(1, 14)] + [1])[::3]

methods = [
            # 'random', 
            # 'random-weight',
            # 'static-weight', 
            # 'nf-llf', 
            'nf-po2', 'heuristic', 
            # 'kf1d'
            ]
feature = 'fd_res_avg_disc'

if __name__ == "__main__":  # confirms that the code is under main function
    tasks = []
    T0 = time.time()
    counter = Value('i', 0)

    root_dir = '../../../data/simulation/normal-server-capacity-variance-test'
    create_path(root_dir)

    for fct_mu in [0.5]:
        for fct_std in [0.15]:
            trace = 'normal-{:.3f}-{:.3f}'.format(fct_mu, fct_std)
            create_path(os.path.join(root_dir, trace)) # create path $root_dir/$trace
            lambda_list = query_rate_list
            for method in methods:
                # create path $root_dir/$trace/$method
                create_path(os.path.join(root_dir, trace, method))
                for poisson_lambda in lambda_list:
                    dir_ = os.path.join(root_dir, trace, method,
                                        '{:.3f}'.format(poisson_lambda))
                    # create path $root_dir/$trace/$method/$traffic_rate
                    create_path(dir_)

                    for capacity_variance in range(1, 10):
                        dir_lb = os.path.join(dir_, 'cv{}'.format(capacity_variance))
                        # create path $root_dir/$trace/$method/$traffic_rate/$cv_variance
                        create_path(dir_lb)
                        if method == 'kf1d': # compare different std pairs
                            for process_std in [0.01]:
                                for sensor_std in [0.05, 0.1, 0.2]:
                                    dir_lb_ = os.path.join(
                                        dir_lb, '{:.2f}-{:.2f}'.format(process_std, sensor_std))

                                    cmd = 'python3 ../run.py -t {0} --fct-mu {1} --lambda {2:.3f} -m {3} -w {4}-reduce-trace.log --n-worker {5} --n-as {6} --n-lb {7} --fct-std {8} --fct-type {9} --kf-sys-std {10} --kf-sensor-std {11} --n-worker-multiplier {12} > {4}-rl.log'.format(
                                        t_episode, fct_mu, poisson_lambda, method, dir_lb_, n_worker, n_as, n_lb, fct_std, 'normal', process_std, sensor_std, capacity_variance)
                                    tasks.append([cmd, dir_lb.lstrip(root_dir)])

                        else:
                            cmd = 'python3 ../run.py -t {0} --fct-mu {1} --lambda {2:.3f} -m {3} -w {4}/reduce-trace.log --n-worker {5} --n-as {6} --n-lb {7} --fct-std {8} --fct-type {9} --n-worker-multiplier {10} > {4}/rl.log'.format(
                                t_episode, fct_mu, poisson_lambda, method, dir_lb, n_worker, n_as, n_lb, fct_std, 'normal', capacity_variance)
                            tasks.append([cmd, dir_lb.lstrip(root_dir)])

    total_task = len(tasks)
    # for t in tasks:
        # print(t)
    print('total task = {}'.format(total_task))
    pool_handler(tuple(tasks))
