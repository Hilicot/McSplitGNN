import argparse
import os
import numpy as np
from src.sort_heuristic import *

class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        dataset_list = [
            ["pair_02/g1.txt","pair_02/g2.txt"]
        ]

        self.parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Specify the device to use.')
        self.parser.add_argument('--dataset_format', type=str, default='binary', choices=['binary', 'ascii'], help='Specify the format of the dataset files.')
        self.parser.add_argument('--data_folder', type=str, default=os.path.join("dataset"), help='path to the folder containing the dataset files')
        self.parser.add_argument('--dataset_list', type=list, default=dataset_list, help='list of datasets to train on')
        self.parser.add_argument('--timeout','-t', type=int, default=np.inf, help='timeout in seconds')
        self.parser.add_argument('--shuffle_input', type=bool, default=False, help='')
        self.parser.add_argument('--batch_size', type=int, default=1, help='')

        
        self.initialized = True
        self.parser.add_argument('--log_level','-ll', type=str, default='info',
                                 help='Specify the logging level to show. Can be "debug", "info", "warning", "error" or "critical".')
        self.parser.add_argument('--log_file','-lf', type=str, default=None,
                                 help='Specify the file to log to. If not specified, the log will be saved to file.')
        self.parser.add_argument('--log_stdout','-ls', type=bool, default=True,
                                 help='Specify whether to log to stdout.')

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        define_constants(self.opt)

        # static options
        self.opt.swap_policy = self.opt.McSPLIT_SD
        self.opt.sort_heuristic = SortPagerank()
        self.opt.reward_policy = RewardPolicy()


        # other fields to use as gloabl variables
        self.start_time = 0

        return self.opt


class RewardPolicy:
    reward_switch_policy_threshold = 0

def define_constants(opt):
    opt.NO_SWAP = 0
    opt.McSPLIT_SD = 1
    opt.McSPLIT_SO = 2

def null_coalescence(current_value, default_value):
    if current_value is None:
        return default_value

    return current_value