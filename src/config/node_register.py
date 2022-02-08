from common.entities import NodeAS, NodeClient, NodeStatelessLB, NodeLB
from policies.rule import NodeLBLSQ, NodeLBSED, NodeLBSRT, NodeLBGSQ, NodeLBActive
from policies.heuristic import NodeHLB, NodeHLBada


# ---------------------------------------------------------------------------- #
#                             Register All Policies                            #
# ---------------------------------------------------------------------------- #

# ---------------------------------- Methods --------------------------------- #

METHODS = [
    'ecmp',  # ECMP
    'weight',  # Static Weight
    'lsq',  # Local shortest queue (LSQ)
    'lsq2',  # Local shortest queue (LSQ) + power-of-2-choices
    # 'weightlsq',  # LSQ
    # 'weightlsq2',  # LSQ + power-of-2-choices
    # 'heuristiclsq',  # LSQ
    # 'heuristiclsq2',  # LSQ + power-of-2-choices
    # 'kf1dlsq',  # LSQ
    # 'kf1dlsq2',  # LSQ + power-of-2-choices
    'sed',  # LSQ
    'sed2',  # LSQ + power-of-2-choices
    # 'heuristiclsq-dev',  # LSQ
    # 'heuristiclsq2-dev',  # LSQ + power-of-2-choices
    'hlb',  # LSQ
    'hlb2',  # LSQ + power-of-2-choices
    'oracle',  # a god-like LB that knows remaining
    'gsq',  # a god-like LB that knows remaining
    'gsq2',  # a god-like LB that knows remaining
    'hlb-ada', # KF1d + LSQ w/ adaptive sensor error
    'active-wcmp',  # KF1d + LSQ w/ adaptive sensor error
]

NODE_MAP = {
    'clt': NodeClient,
    'er': NodeStatelessLB,
    'lb': NodeLB,  # Maglev
    'as': NodeAS,
    # --------------------------------- Policies --------------------------------- #
    'lb-ecmp': NodeLB,  # ECMP
    'lb-weight': NodeLB,  # Static Weight
    'lb-lsq': NodeLBLSQ,  # Local shortest queue (LSQ)
    'lb-lsq2': NodeLBLSQ,  # Local shortest queue (LSQ) + power-of-2-choices
    # 'lb-misconfig': NodeLBMisconf,  # Misconfigured
    # 'lb-heuristic': NodeLBHeuristic,  # Heuristic
    # 'lb-kf1d': NodeLBKF1D,  # 1D Kalman-Filter-Based (KF1D) LB
    # 'lb-weightlsq': NodeLBWeightLSQ,  # weighted LSQ
    # 'lb-weightlsq2': NodeLBWeightLSQ,  # weighted LSQ + power-of-2-choices
    # 'lb-heuristiclsq': NodeLBHeuristicLSQ,  #, NodeLBHeuristicLSQ Heuristic LSQ
    # 'lb-heuristiclsq2': NodeLBHeuristicLSQ,  # Heuristic LSQ + power-of-2-choices
    # 'lb-kf1dlsq': NodeLBKF1DLSQ,  # KF1D LSQ
    # 'lb-kf1dlsq2': NodeLBKF1DLSQ,  # KF1D LSQ + power-of-2-choices
    'lb-sed': NodeLBSED,  # weighted LSQ
    'lb-sed2': NodeLBSED,  # weighted LSQ + power-of-2-choices
    # 'lb-heuristiclsq-dev': NodeLBHeuristicLSQdev,  # Heuristic LSQ
    # 'lb-heuristiclsq2-dev': NodeLBHeuristicLSQdev,  # Heuristic LSQ + power-of-2-choices
    'lb-hlb': NodeHLB,  # KF1D LSQ
    'lb-hlb2': NodeHLB,  # KF1D LSQ + power-of-2-choices
    'lb-oracle': NodeLBSRT,  # KF1D LSQ + power-of-2-choices
    'lb-gsq': NodeLBGSQ,  # Oracle global shortest queue 
    'lb-gsq2': NodeLBGSQ,  # Oracle global shortest queue + power-of-2-choices
    'lb-hlb-ada': NodeHLBada,  # adaptive KF1D LSQ
    'lb-active-wcmp': NodeLBActive,  # active probing KF1D LSQ
}
