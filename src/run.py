from common.simulator import *
from config.user_conf import *

UNIT_TRAFFIC_RATE = 0

def init_global_variables(args):
    global METHODS, LOG_FOLDER, NODE_CONFIG, DEBUG, UNIT_TRAFFIC_RATE

    assert args.method in METHODS, "method {} not in legal METHODS".format(args.method)

    NODE_CONFIG = generate_node_config_tier4(
        lb_method=args.method,
        n_clt=args.n_clt,
        n_er=args.n_er,
        n_lb=args.n_lb,
        n_as=args.n_as,
        n_worker_baseline=args.n_worker,
        n_worker2change=int(args.n_worker2change*args.n_as),
        n_worker_multiplier=args.n_worker_multiplier,
        as_mp_level=args.as_mp_level,
        kf_sensor_std=args.kf_sensor_std,
        lb_bucket_size=args.lb_bucket_size,
        b_offset=args.b_offset,
        lb_period=args.lb_period,
        debug=DEBUG)

    print("args.lb_period={}".format(args.lb_period))

    fct_mu = args.cpu_fct_mu
    if args.process_n_stage > 1:
        fct_mu += args.io_fct_mu
    UNIT_TRAFFIC_RATE = sum([v['n_worker'] for v in NODE_CONFIG['as'].values()]) / fct_mu / args.n_clt
    traffic_rate = args.poisson_lambda * UNIT_TRAFFIC_RATE

    app_config = get_app_config(traffic_rate, *[eval('args.{}'.format(k)) for k in getfullargspec(get_app_config)[0][1:]])

    for i in NODE_CONFIG['clt'].keys():
        NODE_CONFIG['clt'][i].update({'app_config': app_config}),

    # update log folder
    LOG_FOLDER = args.log_folder

    # print out basic info
    print("unit traffic rate for current setup: {}".format(UNIT_TRAFFIC_RATE))

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    init_global_variables(args)

    simulator = Simulator(NODE_CONFIG, CP_EVENTS2ADD,
                          logfolder=LOG_FOLDER, debug=DEBUG)

    simulator.run(args.n_episode, args.t_stop, args.t_inc, args.t_stddev, args.n_flow_total, args.first_episode_id)
