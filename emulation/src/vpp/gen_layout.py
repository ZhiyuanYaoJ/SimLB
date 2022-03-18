#!/usr/bin/python3.6
import sys
import os
# add common utils to path
dirname = os.path.dirname(__file__)
utils_dir = os.path.join(dirname, '../utils')
sys.path.insert(0, utils_dir) 
from common import *

parser = argparse.ArgumentParser(description='Generate shared memory layout header for LB plugin in VPP.')

parser.add_argument('-m', action='store',
                    default='maglev',
                    type=str,
                    dest='method',
                    help='Load Balancing Method')

parser.add_argument('-dn', action='store_true',
                    default=False,
                    dest='debug_node',
                    help='Set True if clib_warning in node.c should be seen')

parser.add_argument('--version', action='version',
                    version='%(prog)s 1.0')

#--- utils ---#

def get_line (var, config, diff=True):
    '''
    @brief:
        put all attributes into one line in the header file
    @param:
        diff: if True then add "_" for array fields
    '''
    res = ""
    if not isinstance(var[2], int):
        var[2] = config['global'][var[2]]
        if diff:
            if var[2] > 1:
                res += "_"
    if var[0] not in global_config["map"]["ctype2byte"].keys():
        var[0] += "_t"
    return res+"_({}, {}, {}, \"{}\", {}) ".format(*var)        

#--- macro ---#

LAYOUT_FILENAME = join(COMMON_CONF['dir']['root'], 'src', 'lb', 'shm_layout.json')

#--- main ---#

if __name__ == '__main__':

    args = parser.parse_args()

    global_config = json_read_file(LAYOUT_FILENAME)
    
    # for different methods, add different macros
    assert (args.method in LB_METHODS.keys())
    lines = LB_METHODS[args.method]['vpp_macro']
    lines.append("")

    for k, v in global_config["global"].items():
        if "_FMT" in k: continue # skip format strings
        lines.append("#define {} {}".format(k, v))
    lines.append("")

    for k, vars in global_config["vpp"]["struct"].items():
        lines.append("#define lb_foreach_{} \\".format(k))
        for var in vars:
            lines.append(get_line(var, global_config))
            if var != vars[-1]:
                lines[-1] += "\\"
        lines.append("")

    lines.append("#define lb_foreach_typedef_struct \\")
    for k, vars in global_config["vpp"]["struct"].items():
        lines.append("_construct({}) \\".format(k))
    lines.append("")

    for k,vars in global_config["vpp"]["enum"].items():
        lines.append("#define lb_foreach_{} \\".format(k))
        for var in vars:
            lines.append("_({}, \"{}\") ".format(*var))
            if var != vars[-1]:
                lines[-1] += "\\"
        lines.append("")

    lines.append("#define lb_foreach_layout \\")
    vars = global_config["layout"]
    for var in vars:
        print(var[0])
        lines.append(get_line(var, global_config, diff=False))
        if var != vars[-1]:
            lines[-1] += "\\"
    lines.append("")

    if args.debug_node:
        lines.append("#define LB_DEBUG_NODE")

    write_file(lines, join(COMMON_CONF['dir']['root'], 'src', 'vpp', 'lb', 'shm.h'))
