import sys
sys.path.insert(0, '..')
import json
from common.events import *
from common.entities import *
from config.global_conf import *
from config.node_register import METHODS, NODE_MAP
from pathlib import Path
from common.alias_method import *

class Simulator:

    global event_buffer

    def __init__(
            self,
            node_config,
            cp_events,
            logfolder='log',
            dump_all_flow=False,
            t_episode=EPISODE_LEN,
            t_episode_inc=EPISODE_LEN_INC,
            n_flow_total=N_FLOW_TOTAL,
            auto_clustering = False,
            debug=0):
        self.node_config = node_config
        self.cp_events = cp_events
        self.logfolder = logfolder
        Path(logfolder).mkdir(parents=True, exist_ok=True)
        self.dump_all_flow = dump_all_flow
        self.auto_clustering = auto_clustering
        self.debug = debug
        self.nodes = {}
        self.t_episode = t_episode
        self.t_episode_inc = t_episode_inc
        self.n_flow_total = n_flow_total
        
        
    def init_nodes(self):
        for node_type, config in self.node_config.items():
            node_type_prefix = node_type
            if 'lb' in node_type: node_type_prefix = 'lb'
            for i, conf in config.items():
                node_id = '{}{}'.format(node_type_prefix, i)
                self.nodes[node_id] = NODE_MAP[node_type](node_id, **conf)
        if self.auto_clustering == True: 
            self.clustering_agent = ClusteringAgent(self.nodes, self.node_config, method = CLUSTERING_METHOD, debug=0)


    def reset(self):
        self.n_flow_done = 0
        self.n_flow_rejected = 0

        event_buffer.reset()
        event_buffer.put(Event(float('inf'), 'end_of_the_world', 'god', {}), checkfull=False)
        for event in self.cp_events: event_buffer.put(Event(*event))
 
        if self.nodes == {}:
            self.init_nodes()
        else:
            for node in self.nodes.values():
                node.reset()

        


    def log_episode(self, episode_id):
        res = {}
        
        
        if self.dump_all_flow:
            flow_all = []

        for node_id, node in self.nodes.items():
            if 'clt' in node_id:
                res[node_id] = node.summarize()
                self.n_flow_rejected += res[node_id]['n_reject']
                if self.dump_all_flow:
                    flow_all += node.flows
            if 'lb' in node_id:
                res[node_id] = node.summarize()
            else:
                continue
        # dump dict into a json
        with open(self.logfolder+'/ep{}.json'.format(episode_id), 'w') as fp:
            json.dump(res, fp)

        if self.dump_all_flow:
            with open(self.logfolder+'/flow{}.json'.format(episode_id), 'w') as fp:
                fp.write('-'*10 + ' episode {} ({} flows total)'.format(episode_id, len(flow_all)) + '-'*10 + '\n')
                for flow_ in flow_all:
                    fp.write("{}\n".format(str(flow_.get_info())))
                    #print(str(flow_.get_info()))
                    if 'reject' in flow_.nexthop:
                        self.n_flow_rejected += 1

    def run_episode(self, episode_id):
        sim_time = 0
        e_next = event_buffer.pop()
        t0 = time.time()
        # t_last = t0
        
        if self.n_flow_total and self.n_flow_total > 0: # prioritize n_flow_total
            eval2run = "self.n_flow_done < self.n_flow_total"
        else:
            eval2run = "sim_time < self.t_episode"
        while eval(eval2run):

            sim_time, event, added_by, kwargs = e_next

            # t1 = time.time()
            
            # if (t1 - t_last > 0.0002):
            #     f = open("WEIGHTS","r")
            #     data = f.read().split("\n")
            #     w = []
            #     for i in data:
            #         if i != "":
            #             w.append(float(i))
            #     update_alias(w)
            #     t_last = time.time()
                        
            if self.debug > 0:
                print('++=======++')
                line = '|| EVENT || ({:.3f}s): {} {}'.format(
                    sim_time, event, kwargs)
                if event == 'dp_receive':
                    flow = kwargs['flow']
                    line += ' {}'.format(flow.get_info())
                print(line)
                print('++=======++')

            arg_keys = getfullargspec(eval(event))[0][2:]
            eval(event)(self.nodes, sim_time, *[kwargs[k] for k in arg_keys])
            e_next = event_buffer.pop()

            if event == 'dp_receive' and kwargs['flow'].t_end: # a flow arrives at client node
                self.n_flow_done += 1
        else:
            msg = '*** [ real {:.3f}s | sim. {:.3f}s ] end of episode {} - {} flows finished total - {} events pending ***'
            print(msg.format(time.time() - t0, sim_time, episode_id, self.n_flow_done, event_buffer.qsize()))

    
        # when the episode finishes
        # 1. we dump all finished flows into a log file if the log file is given
        if self.logfolder:
            self.log_episode(episode_id)
        # 2. we print out necessary data
        if DEBUG > 1:
            print('number of rejected flow in total: {}'.format(self.n_flow_rejected))
            print('-' * 20)

    def run(self, n_episode=N_EPISODE, first_episode_id=0):
        '''
        @brief:
            run $n_episode times episodes where the ${i}-th episode lasts np.random.normal($t_episode + $i * $t_episode_inc, $t_episode_stddev) (clipped by mean +- 2 * stddev)
        @note:
            for now we assume that all episodes share the same t_episode length
        '''
        assert isinstance(n_episode, int) and n_episode > 0, "Number of episodes should be positive integer"

        
        
        # init_alias([1 for i in range(64)]) #wrongfully fixed
        # file = open("WEIGHTS","w")
        # for w in [1 for i in range(64)]:
        #     file.write(str(w) + "\n")
        # file.close()
        
        
        for episode in range(first_episode_id, first_episode_id+n_episode):
            # reset environment
            self.reset()
            
            if not self.n_flow_total or self.n_flow_total < 0:
                # update episode duration (use exponential so that we only need 1 parameter since mean==stddev)
                self.t_episode += np.random.exponential(self.t_episode_inc)

            self.run_episode(episode)
