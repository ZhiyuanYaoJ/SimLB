import json
import matplotlib.pyplot as plt
import numpy as np

fct_data = {}

methods = ['gsq', 'gsq2', 'hermes', 'lsq', 'sed', 'wcmp', 'ecmp']

rates = ['0.610', '0.715', '0.785', '0.845', '0.905', '0.965']

path_tem = "../data/simulation/first-impression/1lb-20as-1worker-1stage-exp-0.50cpumu/"

n_episodes = 5

for meth in methods:
	fct_data[meth] = {}
	for rate in rates:
		ls = []
		for ep in range(n_episodes):
			path = path_tem + "{0}/rate{1}/ep{2}.json".format(meth,rate,ep)
			file = open(path,)
			json_data = json.load(file)
			for i in json_data['clt0']['all_fct']:
				ls.append(i)
			file.close()
			fct_data[meth][rate] = ls


COLORS_DICT = {
    'hermes': "#508fc3", 
    'gsq2': '#559d3f',
    'gsq': '#559d3f',
    'lsq': "#ef8532", 
    'sed': "#ef8532",
    'wcmp': "#0B3191",
    'ecmp': "#8B0B91"
}

MARKER_DICT = {
    'gsq':    "x",
    'gsq2': ".",
    'hermes': "p",
    'lsq':   "X",
    'sed':   "d",
    'wcmp': "x",
    'ecmp': "d"
}

LINESTYLE_DICT = {
    'gsq':   "-.",
    'gsq2':   "-",
    'hermes':       "-",
    'lsq':    "-",
    'sed':   "-",
    'wcmp': "--",
    'ecmp': "--"
}

METHOD_MAPPER = {
    'gsq': "GSQ", 
    'gsq2': "GSQ-PO2",
    'hermes': "HERMES",
    'lsq': "LSQ",
    'sed': "SED",
    'wcmp': "WCMP",
    'ecmp': "ECMP"
}

for rate in rates:
    fig = plt.figure(figsize=(6,2), dpi=96)
    # plot with data as the data source
    for m in methods:
    #     if 'vm-' in m: continue
        c, linestyle, marker = COLORS_DICT[m], LINESTYLE_DICT[m], MARKER_DICT[m]
        data = np.array(fct_data[m][rate])
        percentiles = [np.percentile(data, p) for p in range(0, 110, 10)]
        plt.plot(np.sort(data), np.linspace(0, 1, len(data), endpoint=False), linestyle=linestyle, color=c)
        plt.plot(percentiles, np.linspace(0, 1, 11, endpoint=True), marker, color=c)
        plt.plot(0, -1, linestyle=linestyle, marker=marker, color=c, label=METHOD_MAPPER[m])
    # plt.legend(bbox_to_anchor=(0., 1.05, 2.2, .102), loc='lower left',
    #            ncol=3, mode="expand", borderaxespad=0.)
    plt.grid()
    plt.legend()
    plt.ylabel("CDF")
    plt.xlabel("FCT (s)")
    plt.xlim([0, 6])
    plt.ylim([0, 1])
    # plt.xticks([300, 600, 1000, 1400, 2000, 3000, 4000], [300, 600, 1000, 1400, 2000, 3000, 4000])
    plt.show()
    fig.savefig('/home/crizzi/Desktop/SimLB_v2/src/plots/fct_rate{}.pdf'.format(rate), bbox_inches='tight', transparent=True)
