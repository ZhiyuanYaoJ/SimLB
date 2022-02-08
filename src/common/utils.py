# ---------------------------------------------------------------------------- #
#                                  Description                                 #
# This file contains all utils functions that might be called by other modules #
# ---------------------------------------------------------------------------- #

import math
import random
from random import expovariate
import numpy as np
from numpy.random import uniform, normal, exponential, randint, lognormal
import socket, struct # to translate ip address
from zlib import crc32 # to calculate hash
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import sqeuclidean, jensenshannon
from config.global_conf import *

# ---------------------------------- Traffic --------------------------------- #

def get_poisson_next_time(poisson_lambda):
    '''
    @brief:
        generate delta t for the next event in a Poisson stream
    @params:
        poisson_lambda: lambda as in Poisson arriving rate
    '''
    return -math.log(1.0 - random.random()) / poisson_lambda

def get_fct_exp(mu):
    '''
    @brief:
        generate expected flow complete time (FCT) w/ exponential distribution
    @params:
        mu: average FCT, mu=stddev for exponential distribution
    '''
    return exponential(mu)

def get_fct_normal(mu, std):
    '''
    @brief:
        generate expected flow complete time (FCT) w/ normal distribution clipped by 2*stddev on the left and 3*stddev on the right, which gives us 97.6% datapoints
    @params:
        mu: average FCT, mu
        stddev: standard deviation (should be less than mu/2)
    '''
    assert mu - 2*std > 0
    return np.clip(normal(mu, std), FCT_MIN, float('inf'))

def get_fct_lognormal(mu, std):
    '''
    @brief:
        generate expected flow complete time (FCT) w/ lognormal distribution
    @params:
        mu: average FCT, mu
        stddev: standard deviation
    '''
    sigma = np.sqrt(np.log(std**2 / np.exp(2*np.log(mu)) + 1))
    mu = np.log(mu) - sigma**2/2
    return lognormal(mu, std)

def get_fct_uniform(mu, std):
    '''
    @note:
        the name of arguments are not intuitive, this is for the sake of generalizing arguments
    '''
    assert mu>0 and std>0 and mu-std>0
    return uniform(mu-std, mu+std)

def get_packet_transmission_time(): 
    '''
    @brief:
        generate a random small number uniformly
    TODO - distance and congestion level could be a factor
    '''
    return uniform(1e-5, 1e-3)

##--- Calculation & Basics ---##

def number_split_n_pieces(number, n):
    assert number > 0
    assert n >= 1
    pieces = []
    for idx in range(n-1):
        # Number between 1 and number
        # minus the current total so we don't overshoot
        pieces.append(uniform(0, number-sum(pieces)))
    pieces.append(number-sum(pieces))
    return pieces
    
def calcul_freshness(ts_baseline, tss, base_number=0.9):
    '''
    @brief:
        calculate datapoint freshness based on timestamps
    @params:
        ts_baseline: current timestamp to compare freshness
        tss: a list of timestamps corresponding to each datapoint
        values: feature values
        base_number: should be less than 1
    TODO learnable/adaptive base number (0.9)
    '''
    if len(tss) > 0:
        assert ts_baseline >= max(tss)
        return np.power(base_number, ts_baseline - np.array(tss))
    else:
        return np.array([])


def calcul_distance(x1, x2):
    assert len(x1) == len(x2)
    res = {
        'rmse': sqeuclidean(x1, x2)/len(x1),
        'js': jensenshannon(x1, x2),
        'wasserstein': wasserstein_distance(x1, x2),
        'ks': ks_2samp(x1, x2), # return (ks-value, pvalue)
    }
    return res

def count_n_interval(lr_pairs, granularity=LOAD_OBSERVE_INTERVAL):
    '''
    @brief:
        count number of intervals for each step
    @return:
        freq: frequency array
        minimum: lowest index
        maximum: highest index
    '''
    assert granularity > 0
    maximum = 0
    minimum = float("inf")
    bin_size = 65536 # a number that is big enough to store all the 
    freq = np.zeros(bin_size)
    for l, r in lr_pairs:
        l_int = int(l/granularity)
        r_int = int(r/granularity) + 1
        freq[l_int] += 1
        freq[r_int] -= 1
        if l_int < minimum: minimum = l_int
        if r_int > maximum: maximum = r_int
    
    for i in range(minimum, maximum):
        freq[i] += freq[i-1]
    return freq, minimum+1, maximum # remove tail, head is kept so that it will be aligned for all AS nodes

def reduce_load(lr_pairs_dict, granularity=LOAD_OBSERVE_INTERVAL):
    freq_dict, max_idx, min_idx = {}, 0, float("inf")
    mins, maxs = [], []
    for as_node, lr_pairs in lr_pairs_dict.items():
        freq, _min, _max = count_n_interval(lr_pairs, granularity=granularity)
        freq_dict[as_node] = freq
        mins.append(_min)
        maxs.append(_max)
    min_idx = int(np.percentile(mins, 0.01))
    max_idx = int(np.percentile(maxs, 0.99))
    assert(min_idx < max_idx)  # overlapping indexes exist
    fairness, overprovision = np.empty(0), np.empty(0)
    qlen_all = []
    for i in range(min_idx, max_idx):
        loads = [v[i] for v in freq_dict.values()]
        qlen_all.append(loads)
        fairness = np.append(fairness, calcul_fair(loads))
        overprovision = np.append(overprovision, calcul_over(loads))
    res = {
        'fairness-avg': fairness.mean(),
        'fairness-std': fairness.std(),
        'over-avg': overprovision.mean(),
        'over-std': overprovision.std(),
        'qlen-all': qlen_all
    }
    return res

def reduce_fct(prefix, data, data_compare):
    res = calcul_distance(data, data_compare)
    res = {'-'.join([prefix,k]): v for k, v in res.items()}
    res[prefix+'-avg'] = data.mean()
    res[prefix+'-std'] = data.std()
    for i in range(0, 110, 10):
        res[prefix+'-p{}'.format(i)] = np.percentile(data, i)
    res[prefix+'-p95'] = np.percentile(data, 95)
    res[prefix+'-p99'] = np.percentile(data, 99)
    return res


def calcul_fair(values):
    '''
    @brief:
        calculate fairness
    @params:
        values: a list of values
    '''
    values = np.array(values)
    n = len(values)
    if sum(values) != 0.:
        return pow(sum(values), 2)/(n*sum(pow(values, 2)))
    else:
        return 1.

def calcul_over(values):
    '''
    @brief: 
        calculate over-provision factor
    @params:
        values: a list of values
    '''
    values = np.array(values)
    return values.max() / (values.mean() + 1e-6)

def get_t_v_from_reservoir_buffer(buffer):
    tss = np.array([e[0] for e in buffer])
    values = np.array([e[1] for e in buffer])
    return tss, values


def softmax(x):
    '''
    Compute softmax values given sets of scores in x.
    '''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

if LB_UPDATE_PADDING == 'valid':
    def get_next_lb_gen_time(last_gen_time, ts, period):
        return ts + period
elif LB_UPDATE_PADDING == 'same':
    def get_next_lb_gen_time(last_gen_time, ts, period):
        return max(last_gen_time+period, ts+1e6) # use max just in case generation takes longer than period, then generate new weights right away
else:
    raise ValueError

def ip2int(addr):
    return struct.unpack("!I", socket.inet_aton(addr))[0]

def int2ip(addr):
    return socket.inet_ntoa(struct.pack("!I", addr))


def check_ip4(ip):
    fields = [int(f) for f in ip.split('.')]
    assert len(fields) == 4
    for i, f in enumerate(fields):
        assert f >= 0 and f <= 255
    if fields[-1] == 0:
        fields[-1] = random.randint(1, 255)
    return '.'.join([str(f) for f in fields])


def generate_ip_random(ip_prefix, ip_mask, count=1):
    '''
    @params:
        ip_prefix: (str) e.g. 192.168.1.0
        ip_mask: (int) e.g. 24 ((0, 32])
    @return:
        ip (str)
    '''
    check_ip4(ip_prefix)
    prefix = ip2int(ip_prefix) & (0xffffffff << (32-ip_mask))
    suffix = randint(0, 0xffffffff >> ip_mask, count)
    ip = suffix + prefix
    return ip

def generate_port_random(count=1):
    return randint(1, 0xffff, count)

def hash_5tuple(src_ip, dst_ip, protocol, src_port, dst_port):
    crc = 0
    crc = crc32(bytes(crc), src_ip)
    crc = crc32(bytes(crc), dst_ip)
    crc = crc32(bytes(crc), protocol)
    crc = crc32(bytes(crc), src_port)
    crc = crc32(bytes(crc), dst_port)
    return crc

def hash_2tuple(src_ip, src_port):
    crc = 0
    crc = crc32(bytes(crc), src_ip)
    crc = crc32(bytes(crc), src_port)
    return crc

def ecmp(src_ip, dst_ip, protocol, src_port, dst_port, bucket_table, bucket_mask):
    hash_ = hash_5tuple(src_ip, dst_ip, protocol, src_port, dst_port)
    index = hash_ & bucket_mask
    return index, bucket_table[index]

def ecmp_simple(src_ip, src_port, bucket_table, bucket_mask):
    hash_ = hash_2tuple(src_ip, src_port)
    index = hash_ & bucket_mask
    return bucket_table[index]

def ecmp_random(bucket_table, bucket_mask):
    index = random.choice(range(bucket_mask+1))
    return index, bucket_table[index]


ecmp_methods = {
    False: ecmp_random,
    2: ecmp_simple,
    5: ecmp
}

#--- reward ---#
reward_options = {
    # option 0: 1-overprovision
    0: lambda x: 1 - calcul_over(x),
    # option 1: difference between min and max
    1: lambda x: min(x) - max(x),
    # option 2: Jain's fairness index
    2: lambda x: calcul_fair(x),
}

def fct_generator(fct_type): 
    return eval('get_fct_{}'.format(fct_type))
