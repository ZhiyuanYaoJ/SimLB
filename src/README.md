# Outline

- [Entities](#entities)
- [Events](#events)
- [Policies](#policy)
- [Configuration](#configuration)

# Entities<a name="entities"></a>

- [Event](#event)
- [Flow](#flow)
- [DistributionFCT](#distribution)
- [Application](#application)

## Event<a name="event"></a>

This simulator is driven by events, each as a tuple consisting of `time`, `name`, `added_by`, and `kwargs`.

## Flow<a name="flow"></a>

This class helps instantiate network flows with key informations including timestamps, traversed nodes, *etc*.
Relevant information (*e.g.* timestamps, nexthop, residual processing time) will be updated on reception/forward events.

### Attributes:

- `id`: format=`“{src_node}-#”`, e.g. `“clt0-42”`
- `src_node`: sent from whom
- `t_begin`: timestamp of departure from `src_node`
- `fct`: a list of processing time, each defines the processing time of one stage
- `fct_type`: in `['cpu', 'io]`, define the first stage, if `fct_type='cpu'`, then `fct` represents: [cpu_time0, io_time0, cpu_time1, io_time1, cpu_time2, ...]
- `fields`: header information stored in a dict format, e.g. 2/5-tuple info can be given here if hash needs to be calculated

### Local variables:

- `t_rest`: residual time for current processing stage, initialized as `fct[0]`
- `t_end`: timestamp of reply reception by `src_node`, initialized as `None`
- `nexthop`: denotes the next node to be forwarded to, if the flow is rejected, `reject` will be appended in the `nexthop` string
- `path`: a list of nodes that the flow traverses
- `tss`: a list of timestamps of entering/exiting nodes, e.g. `[ts_enter_middlebox0, ts_exit_middlebox0, ts_enter_middlebox1, ts_exit_middlebox1, ts_enter_middlebox2, ...]`
- `note`: a dict of info that can be added by any nodes the flow traverses

### Functions

- `get_info`: dump all the info of the flow into one line, e.g. `Flow (clt0-8) | clt0[0.072]->[0.072]er0[0.072]->[0.073]lb3[0.073]->[0.073]as66[0.076]->[0.076]clt0 (FCT=0.002s, PLT=0.004s) | FCT: [0.00236792] | {}`, where page load time (PLT) = flow completion time (FCT) + transition time, if the flow is rejected, `(rejected)` will appear in the string
- [TBD]

## DistributionFCT<a name="distribution"></a>

Four types of FCT distributions are implemented: 
- `"exp"`: exponential distribution, defined by `"mu"`,
- `"normal"`: Gaussian distribution, defined by `"mu"` and `"std"`,
- `"lognormal"`: Lognormal distribution, defined by `"mu"` and `"std"`,
- `"uniform"`: uniform distribution, for simplicity (not adding more global arguments), defined by `"mu"` and `"std"`, which then helps calculate `min=mu-std` and `max=mu+std`

Once the type of the distribution is initialized, it does not change any more.

### Functions

- `get_value`: a callback function defined in `utils.py`, corresponding to the type of distribution, will be executed to generate a scaler.

## Application<a name="application"></a>


# Events<a name="events"></a>


# Policies<a name="policy"></a>



# Configuration<a name="configuration"></a>

## Base Json

The configuration stored in `master.json` is the base configuration for all the simulations. Branches can be created by updating fields in the dictionary.

- `experiment`: name of the experiment, a folder of this name will be created in `data` directory to store all the info
- `seed`: random seed
- `nodes`
  - `as`: for all application servers
    -  `n`: number of AS nodes
          -  `isinstance(n, int)`
          -  `n > 0`
    -  `n_max`: maximum number of AS nodes, aka action space dim for RL agents 
          -  `n_max >= n`
    -  `n_cpu_base`: number of cpu workers for all ASes as a baseline 
          -  `isinstance(n_cpu_base, int)`
          -  `n_cpu_base >= 1`
    -  `mp_level`: each cpu worker can process how many jobs at the same time
          -  `isinstance(mp_level, int)`
          -  `mp_level >= 1`
    -  `cpu_upgrade_ratio`: ratio of ASes that have upgraded processing capacity
          -  `cpu_upgrade_ratio >= 0 and cpu_upgrade_ratio < 1`
    -  `cpu_upgrade_value`: how much do we upgrade AS processing capacity (1->no upgrade)
          -  `cpu_upgrade_value > 1`
          -  `int(cpu_upgrade_value*n_cpu_base) == cpu_upgrade_value*n_cpu_base`
    - `n_io`: number of io workers for all ASes
        - `n_io >= 1`
    - `backlog`: maximum allowed number of ongoing connections, reject new flows if the sum of all the queue lengths exceed this number 
        - `isinstance(backlog, int)`
        - `backlog >= n_cpu_base*cpu_upgrade_value*mp_level`
    - `timeout`: rejected flows will be returned to client after 40 seconds
    - `debug`: level of debug message (0: no debug message, ..., 2: all debug messages)
  - `er`: for all edge routers
    - `n`: number of edge routers
          -  `isinstance(n, int)`
          -  `n > 0`
  - `lb`: for all load balancers
    - `n`: number of LB nodes
          -  `isinstance(n, int)`
          -  `n > 0`
    - `update_period`: time intervals between two consecutive weights updates, unit is second
        - `update_period > 0`
    - `update_padding`: 
      - 'valid': keep incrementing `update_period` on last weights generation timestamp
      - 'same': keep intervals between weights generating timestamps the same as `update_period`
      - `update_padding in ['valid', 'same']`
    - `method`: name of the load distribution method, should be within the list in `node_register.py`
    - `n_bucket`: bucket size for the flow table
      - `isinstance(n_bucket, int)`
      - `n_bucket > 128`
    - `render`: render information into log file if set true
      - `step`: render whenever LB agent calls `step` function
      - `receive`: render whenever LB agent calls `receive` function
    - `rtt`: rtt between LB and AS
      - `min`: `>0`
      - `max`: `>min`
  - `clt`: for all clients
    - `n`: number of clients
          -  `isinstance(n, int)`
          -  `n > 0`
    - `gt_period`: dump ground truth every `gt_period`, unit is second
      - `gt_period > 0`
- `feature`: networking features that we collect
  - `reservoir`: features collected using reservoir sampling
    - `fresh_base`: exponential base number to calculate freshness in utils
      - `fresh_base > 0 and fresh_base < 1`
    - `size`: buffer size
      - `isinstance(size, int)`
      - `size > 16`
    - `p`: probability of replacing an old sample by a new one
      - `p > 0 and p <= 1`
    - `reduce_option`: a list of process methods to reduce the samples into one scalar
      - 'avg',          # simply calculate average
      - 'std',          # simply calculate standard deviation
      - 'p90',          # 90th-percentile
      - 'avg_disc',     # discounted weighted averaged based on samples' freshness
      - 'avg_decay',    # simple average based on samples' freshness
  - `as`: feature for application servers
    - `counter`: a list of counters
      - [TBD] implemented where
    - `reservoir`: a list of features collected 
      - [TBD] implemented where
  - `lb`: feature for load balancers
    - `counter`: a list of counters
      - [TBD] implemented where
    - `reservoir`: a list of features collected 
      - [TBD] implemented where
- `simulation`:
  - `n_episode`: number of episodes to run
    - `isinstance(n_episode, int)`
    - `n_episode > 0`
  - `first_episode`: id of the first episode
    - `isinstance(first_episode, int)`
    - `first_episode > 0`
  - `t_len`: (unit second) total simulation time
    - `t_len > 0`
  - `n_flow`: total number of flows to run
    - `isinstance(n_flow, int)`
    - if `n_flow > 0`, ignore `t_len` and stop simulation once we reach `n_flow`
    - otherwise, we use `t_len` to define simulation duration
  - `t_inc`: incremental episode length, next episode will be longer by `np.random.exponential(t_inc)`
    - `t_inc > 0`
  - `log`: dump results to this log file if not `null`
  - `dump_all_flow`: set to `True` if we want to log all the flows' info line by line
  - `debug`: level of debug message (0: no debug message, ..., 2: all debug messages)
- `traffic`:
  - `rate`: normalized traffic rate, need to multiple average FCT and the number of ASes
    - `rate > 0`
  - `n_stage`: number of processing stages
    - `isinstance(n_stage, int)`
    - `n_stage >= 1`
  - `first_stage`: either `cpu` or `io`
    - `first_stage in ['cpu', 'io']`
  - `cpu_fct` and `io_fct`: information about cpu/io processing time distribution
    - `type`: distribution type
      - `type in ['exp', 'normal', 'lognormal', 'uniform']`
    - `mu`: mean
      - `mu > 0`
    - `std`: stddev
      - `std > 0`
      - `if type=='normal': assert mu - 2*std > 0`
    - `min`: min
      - `min > 0`
    - `max`: max
      - `max > min`
- `policy`:
  - `alpha`: soft update alpha
    - `alpha >= 0 and alpha <= 1`
  - `beta`: offset for SED
    - `isinstance(beta, int)`
    - `beta >=1`
  - `kf`: Kalman filter config
    - `system_std`: `>=0`
    - `sensor_std`: `>=0`,
    - `system_mean`: `>=0`,
    - `init_mean`: `>=0`,
    - `init_std`: `>=0`
  - `reward`: define the key of reward function
    - [TBD]
- `cp_event`: control plane events (defined in `events.py`) [TBD]
  - with the following format (ts, event_name, added_by, **kwargs)
  - e.g. change some ASes processing capacities back to normal worker baseline:
    - `[6., "as_update_capacity", "sys-admin", {"node_ids": ["as{}".format(i) for i in range(32, 64)], "n_worker": 5, "mp_level": 2}]`
  - e.g. remove some ASes from one LB's children nodes
    - `[9.6, "lb_remove_server", "sys-admin", {"lbs": [0], "ass": [17,18]}]`
  - or update traffic distribution...

# TODO

- Replace `RESERVOIR_FD_PACKET_DENSITY` by a value calculated using traffic rate
- Keep `REDUCE_METHODS`, `RESERVOIR_AS_KEYS`, `RESERVOIR_LB_KEYS`
- Rewrite `REWARD` functions