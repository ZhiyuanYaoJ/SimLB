import logging
import sys
import datetime
import numpy as np
import time
import os
from os.path import join, isfile
import json
import argparse

def init_logger(filename, logger_name):
    '''
    @brief:
        initialize logger that redirect info to a file just in case we lost connection to the notebook
    @params:
        filename: to which file should we log all the info
        logger_name: an alias to the logger
    '''

    # get current timestamp
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
    
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s] %(name)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename=filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Test
    logger = logging.getLogger(logger_name)
    logger.info('### Init. Logger {} ###'.format(logger_name))

    return logger

def json_write2file(data, filename):
    '''
    @brief:
        write data to json file
    '''
    with open(filename, "w") as write_file:
        json.dump(data, write_file, indent=4) # w/ indent the file looks better

def json_read_file(filename):
    '''
    @brief:
        read data from json file
    '''
    with open(filename, "r") as read_file:
        return json.load(read_file)

def write_file(lines, filename):
    '''
    @desc:    
        write lines from a list of strings to a file
    @params:  
        (str) filename
        (list[str]) lines
    '''
    with open (filename, "w") as f:
        for l in lines:
            f.write("{}\n".format(l))

def create_folder(_dir):
    '''
    @brief:
        create folder if does not exist
    '''
    for d in _dir:
        if not os.path.exists(d):
            os.mkdir(d)

# @desc:   recursively get files given a directory
# @params: (str)path
# @return: list of directories
def get_files_r(path):
    files = []
    for root, directories, filenames in os.walk(path):
        for filename in filenames: 
            files.append(os.path.join(root,filename))
    return [f for f in files if '/.' not in f]

# @desc:   read lines from a file into a list
# @params: (str)filename
# @return: list of strings
def read_file(filename):
    lines = []
    with open (filename, "r") as myfile:
        lines = [line.rstrip('\n') for line in myfile]
    return lines

# @desc:   read csv file into pandas dataframe
# @params: filename
# @return: pandas dataframe
def pd_dataframe_from_csv(filename, sep=','):
    return pd.read_csv(filename, sep=sep, encoding='utf-8')

dirname = os.path.dirname(__file__)
# get common configuration
filename = os.path.join(dirname, '../../global_conf.json')
COMMON_CONF = json_read_file(filename)
# get all LB methods
filename = os.path.join(dirname, '../../config/lb-methods.json')
LB_METHODS = json_read_file(filename)