import argparse
import os
import numpy as np
from src.sort_heuristic import *
import copy


class BaseOptions:
    parse_opt = None
    opt = None

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialized = False

    def initialize(self):
        dataset_list = [
            ["gl_search/pair_3_graphs_0_1/g1.txt", "gl_search/pair_3_graphs_0_1/g2.txt"]
        ]

        self.parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            choices=["cpu", "cuda"],
            help="Specify the device to use.",
        )
        self.parser.add_argument(
            "--dataset_format",
            type=str,
            default="binary",
            choices=["binary", "ascii"],
            help="Specify the format of the dataset files.",
        )
        self.parser.add_argument(
            "--data_folder",
            type=str,
            default=os.path.join("dataset"),
            help="path to the folder containing the dataset files",
        )
        self.parser.add_argument(
            "--dataset_list",
            type=list,
            default=dataset_list,
            help="list of datasets to train on",
        )
        self.parser.add_argument(
            "--timeout", "-t", type=int, default=np.inf, help="timeout in seconds"
        )
        self.parser.add_argument(
            "--max_iter",
            "-i",
            type=int,
            default=np.inf,
            help="max number of iterations",
        )
        self.parser.add_argument("--quiet", "-q", action="store_true", help="")
        self.parser.add_argument("--shuffle_input", type=bool, default=False, help="")
        self.parser.add_argument("--batch_size", type=int, default=1, help="")

        self.initialized = True
        self.parser.add_argument(
            "--log_level",
            "-ll",
            type=str,
            default="info",
            help='Specify the logging level to show. Can be "debug", "info", "warning", "error" or "critical".',
        )
        self.parser.add_argument(
            "--log_file",
            "-lf",
            type=str,
            default=None,
            help="Specify the file to log to. If not specified, the log will be saved to file.",
        )
        self.parser.add_argument(
            "--log_stdout",
            "-ls",
            type=bool,
            default=True,
            help="Specify whether to log to stdout.",
        )

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.parse_opt = self.parser.parse_args()
        self.opt = Options(self.parse_opt)
        return self.opt


class Options:
    def __init__(self, parse_opt):
        # copy all the options from the parse options
        for k, v in parse_opt.__dict__.items():
            self.__dict__[k] = copy.deepcopy(v)

        # define constants
        self.c = Constants()

        # static options
        self.swap_policy = self.c.McSPLIT_SD
        self.sort_heuristic = SortDegree()
        self.reward_policy = RewardPolicy(self.c)
        self.mcs_method = self.c.RL_DAL


class Constants:
    NO_SWAP = 0
    McSPLIT_SD = 1
    McSPLIT_SO = 2
    RL_POLICY = 10
    DAL_POLICY = 12
    NO_CHANGE = 20
    CHANGE = 21
    RESET = 22
    RANDOM = 23
    STEAL = 24
    RL_DAL = 30
    LL_DAL = 31


class RewardPolicy:
    def __init__(self, c: Constants):
        self.reward_switch_policy_threshold = 0
        self.reward_policies = [c.RL_POLICY, c.DAL_POLICY]
        self.reward_policies_num = len(self.reward_policies)
        self.policy_switch_counter = 0
        self.switch_policy = c.CHANGE
        self.current_reward_policy = 1


def null_coalescence(current_value, default_value):
    if current_value is None:
        return default_value

    return current_value