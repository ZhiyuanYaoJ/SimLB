from common.simulator import *
from config.user_conf import *

UNIT_TRAFFIC_RATE = 0


def init_global_variables(args):
    '''
    @brief: initialize configuration for all the node
    '''

    assert args.method in METHODS.keys(), "method {} not in legal METHODS".format(args.method)

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
        log_folder=args.log_folder,
        rl_test=args.rl_test,
        debug=DEBUG)


    fct_mu = args.cpu_fct_mu
    if args.process_n_stage > 1:
        fct_mu += args.io_fct_mu
    UNIT_TRAFFIC_RATE = sum([v['n_worker'] for v in NODE_CONFIG['as'].values()]) / fct_mu / args.n_clt
    traffic_rate = args.poisson_lambda * UNIT_TRAFFIC_RATE

    app_config = get_app_config(traffic_rate, *[eval('args.{}'.format(k)) for k in getfullargspec(get_app_config)[0][1:]])

    for i in NODE_CONFIG['clt'].keys():
        NODE_CONFIG['clt'][i].update({'app_config': app_config}),

    # update log folder
    global LOG_FOLDER 
    LOG_FOLDER= args.log_folder


    # print out basic info
    print("unit traffic rate for current setup: {}".format(UNIT_TRAFFIC_RATE))

    return NODE_CONFIG

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    NODE_CONFIG = init_global_variables(args)

    simulator = Simulator(
                    NODE_CONFIG,
                    CP_EVENTS2ADD,
                    logfolder=LOG_FOLDER,
                    dump_all_flow=args.dump_all_flow,
                    t_episode=args.t_stop,
                    t_episode_inc=args.t_inc,
                    n_flow_total=args.n_flow_total,
                    debug=DEBUG)

    simulator.run(args.n_episode, args.first_episode_id)
