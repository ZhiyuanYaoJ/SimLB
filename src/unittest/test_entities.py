'''
This file conducts unit-test on src/utils/common.py
'''
import os
import sys
import json
import shutil
import unittest
DIRNAME = os.path.dirname(__file__)
TARGET_DIR = '{}/../common'.format(DIRNAME)
SRC_DIR = '{}/..'.format(DIRNAME)
ROOT_DIR = os.path.join(DIRNAME, '..', '..')
TMP_DIR = os.path.join(ROOT_DIR, 'data', 'tmp')
sys.path.insert(0, TARGET_DIR)
sys.path.insert(0, SRC_DIR)
from entities import Node

class TestUtils(unittest.TestCase):
    '''
    This class tests the src/common/entities.py
    '''

    def test_node_get_t_normal(self):
        '''
        @brief: test get_t2neighbour and get_process_delay with normal range
        '''
        process_delay_uniform_range = (1, 2)
        t2neighbour_uniform_range = (10, 20)
        node2test = Node(
            id='test_node', 
            process_delay_uniform_range=process_delay_uniform_range,
            t2neighbour_uniform_range=t2neighbour_uniform_range)
        process_delay = node2test.get_process_delay()
        t2neighbour = node2test.get_t2neighbour()
        self.assertTrue(process_delay < process_delay_uniform_range[1])
        self.assertTrue(process_delay > process_delay_uniform_range[0])
        self.assertTrue(t2neighbour < t2neighbour_uniform_range[1])
        self.assertTrue(t2neighbour > t2neighbour_uniform_range[0])

    def test_node_get_t_abnormal(self):
        '''
        @brief: test get_t2neighbour and get_process_delay
        '''
        process_delay_uniform_range = (2, 1)
        t2neighbour_uniform_range = (10, 10)
        node2test = Node(
            id='test_node', 
            process_delay_uniform_range=process_delay_uniform_range,
            t2neighbour_uniform_range=t2neighbour_uniform_range)
        process_delay = node2test.get_process_delay()
        t2neighbour = node2test.get_t2neighbour()
        self.assertEqual(process_delay, 0)
        self.assertEqual(t2neighbour, 10)
        