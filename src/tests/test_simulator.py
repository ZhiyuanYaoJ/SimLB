# ---------------------------------------------------------------------------- #
#                                  Description                                 #
# This file aims at testing simulator functionalities                          #
# ---------------------------------------------------------------------------- #


import sys
sys.path.insert(0, '..')
from common.simulator import *
from config.tier_4_lsq import NODE_CONFIG, CP_EVENTS2ADD

simulator = Simulator(NODE_CONFIG, CP_EVENTS2ADD, logfolder='../log/test_lsq', debug=2)

simulator.run(2, 10)