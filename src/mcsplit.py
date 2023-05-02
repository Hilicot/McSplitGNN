from __future__ import annotations
from options import opt
from src.vertex_pair import VertexPair
from src.graph import *
from src.reward import Reward, DoubleQRewards
from src.mcs import mcs
import logging


def mcsplit():
    for [g0_filename, g1_filename] in opt.dataset_list:
        g0 = readGraph(g0_filename)
        g1 = readGraph(g1_filename)

        opt.reward_policy.reward_switch_policy_threshold = 2*min(g0.n, g1.n)

        # decide whether to swap the graphs based on swap_policy
        if do_swap_graphs(g0, g1):
            [g0, g1] = [g1, g0]
            logging.debug("Swapped graphs")

        if True:
            g0_degree = opt.sort_heuristic.sort(g0)
            g1_degree = opt.sort_heuristic.sort(g1)
            g0_sorted = induced_subgraph(g0, g0_degree)
            g1_sorted = induced_subgraph(g1, g1_degree)

            g0_sorted.pack_leaves()
            g1_sorted.pack_leaves()
        else:
            logging.warning("Sorting disabled")
            g0_sorted = g0
            g1_sorted = g1

        rewards = DoubleQRewards(g0_sorted.n, g1_sorted.n)

        solution = mcs(g0_sorted, g1_sorted, rewards)
        
        if not check_sol(g0_sorted, g1_sorted, solution):
            logging.error("Found invalid solution!")
        else:
            pass
        if True:
            logging.info(f"Solution size: {len(solution)}")
            logging.debug(f'Solution: {" ".join([str(pair) for pair in solution])}')

    logging.info("Arguments:")
    logging.info(vars(opt))


def do_swap_graphs(g0, g1):
    if opt.swap_policy == opt.c.McSPLIT_SD:
        # get densities
        d0 = g0.computeDensity()
        d1 = g1.computeDensity()
        # compute density extremeness
        de0 = abs(0.5 - d0)
        de1 = abs(0.5 - d1)
        return de1 > de0
    elif opt.swap_policy == opt.c.McSPLIT_SO:
        return g1.n > g0.n
    elif opt.swap_policy == opt.c.NO_SWAP:
        return False
    else:
        logging.error("swap policy unknown")
        return False

def check_sol(g0: Graph, g1: Graph, solution: List[VertexPair]) -> bool:
    used_left = [False] * g0.n
    used_right = [False] * g1.n
    
    for i in range(len(solution)):
        pair = solution[i]
        
        if used_left[pair.v] or used_right[pair.w]:
            return False
        
        used_left[pair.v] = True
        used_right[pair.w] = True
        
        if g0.adjlist[pair.v].label != g1.adjlist[pair.w].label:
            return False
        
        for j in range(i + 1, len(solution)):
            new_pair = solution[j]
        
            if g0.get(pair.v, new_pair.v) != g1.get(pair.w, new_pair.w):
                return False
    
    return True
