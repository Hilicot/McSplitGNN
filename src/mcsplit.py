from __future__ import annotations
from options import opt
from src.vertex_pair import VertexPair
from src.graph import *
import time


def mcsplit():
    for [g0_filename, g1_filename] in opt.dataset_list:
        g0 = readGraph(g0_filename)
        g1 = readGraph(g1_filename)

        opt.reward_policy.reward_switch_policy_threshold = 2*min(g0.n, g1.n)

        # decide whether to swap the graphs based on swap_policy
        if do_swap_graphs(g0, g1):
            [g0, g1] = [g1, g0]
            print("Swapped graphs")

        g0_degree = opt.sort_heuristic.sort(g0)
        g1_degree = opt.sort_heuristic.sort(g1)

        g0_sorted = induced_subgraph(g0, g0_degree)
        g1_sorted = induced_subgraph(g1, g1_degree)

        g0_sorted.pack_leaves()
        g1_sorted.pack_leaves()

        # TODO init rewards

        opt.start_time = time.time()
        solution = mcs(g0, g1)

        print("Solution size: ", len(solution))

    print("Arguments:")
    print(opt)


def do_swap_graphs(g0, g1):
    if opt.swap_policy == opt.McSPLIT_SD:
        # get densities
        d0 = g0.computeDensity()
        d1 = g1.computeDensity()
        # compute density extremeness
        de0 = abs(0.5 - d0)
        de1 = abs(0.5 - d1)
        return de1 > de0
    elif opt.swap_policy == opt.McSPLIT_SO:
        return g1.n > g0.n
    elif opt.swap_policy == opt.NO_SWAP:
        return False
    else:
        print("swap policy unknown")
        return False
