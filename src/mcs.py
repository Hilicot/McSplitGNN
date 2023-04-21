from __future__ import annotations
from vertex_pair import VertexPair
from reward import Reward, DoubleQRewards
from graph import Graph
import time
import numpy as np


def mcs(g0: Graph, g1: Graph, rewards: DoubleQRewards) -> List[VertexPair]:
    left: List[int] = []  # the buffer of vertex indices for the left partitions
    right: List[int] = []  # the buffer of vertex indices for the right partitions
    domains = []

    left_labels = set()
    right_labels = set()
    for n in g0.adjlist:
        left_labels.add(n.label)
    for n in g1.adjlist:
        right_labels.add(n.label)
    labels = left_labels.intersection(right_labels)  # labels that appear in both graphs

    # Create a bidomain for each label that appears in both graphs
    for label in labels:
        start_l = len(left)
        start_r = len(right)

        for i in range(g0.n):
            if g0.adjlist[i].label == label:
                left.append(i)
        for i in range(g1.n):
            if g1.adjlist[i].label == label:
                right.append(i)

        left_len = len(left) - start_l
        right_len = len(right) - start_r
        domains.append(Bidomain(start_l, start_r, left_len, right_len, False))

    incumbent = [(-1, -1)]*arguments.prime

    solve(g0, g1, incumbent, domains, left, right)

    test_info.recursions = nodes
    return incumbent


def solve(g0: Graph, g1: Graph, best_sol: List[VertexPair], starting_bidomain: List[Bidomain], left: List[int],
          right: List[int]):
    start = time.monotonic()
    max_size = max(g0.n, g1.n)*2
    depth = 0
    current_sol: List[VertexPair] = []
    current_bidomain = [0]*max_size
    bidomains: List[List[Bidomain]] = [starting_bidomain]
    wselected = [False]*g1.n

    v = np.inf
    w = -1

    nodes = 0
    iteration = 1

    while depth >= 0:
        if abort_due_to_timeout(start):
            print("Timeout")
            return

        if opt.max_iter and iteration >= opt.max_iter:
            print("Reached", iteration, "iterations and ", nodes, "v iterations")
            return

        if depth%2 == 0:
            nodes += 1
            bound = len(current_sol) + calc_bound(bidomains[depth//2])

            if bound <= len(best_sol):
                depth -= 1
                if depth < 0:
                    continue
                v, w = current_sol[-1].get()
                current_sol.pop()
                bidomains.pop()
                bidomains[depth//2][current_bidomain[depth//2]].right_len += 1
                continue

            w = -1
            current_bidomain[depth//2] = select_bidomain(bidomains[depth//2], left, len(current_sol))
            if current_bidomain[depth//2] == np.inf:
                depth -= 1
                v, w = current_sol[-1].get()
                current_sol.pop()
                bidomains.pop()
                bidomains[depth//2][current_bidomain[depth//2]].right_len += 1
                continue
            v = solve_first_graph(left, bidomains[depth//2][current_bidomain[depth//2]])
            wselected = [False]*g1.n
            iter += 1
            depth += 1
        else:
            w = solve_second_graph(right, bidomains[depth//2][current_bidomain[depth//2]], wselected)
            iter += 1
            if w != -1:
                current_sol.append(VertexPair(v, w))

                if len(current_sol) > len(best_sol):
                    best_sol = current_sol
                    lap = time.monotonic()
                    time_elapsed = lap - start

                    if not opt.quiet and (nodes - len(best_sol) > 10 or len(best_sol)%100 == 0):
                        print("Incumbent size:", len(best_sol), "Iterations:", iteration, "Time:", time_elapsed)

                bidomains.append(filter_domains(bidomains[depth//2], left, right, g0, g1, v, w,
                                                arguments.directed or arguments.edge_labelled))

                depth += 1
            else:
                bidomains[depth//2][current_bidomain[depth//2]].right_len += 1
                depth -= 1

                if bidomains[depth//2][current_bidomain[depth//2]].left_len <= 0:
                    remove_bidomain(bidomains[depth//2], current_bidomain[depth//2])


def solve_first_graph(nodes: List[int], bd: Bidomain) -> int:
    end = bd.l + bd.left_len
    idx = selectV_index(nodes, rewards, bd.l, bd.left_len)
    # put vertex at the back
    if idx != np.inf:
        nodes[idx], nodes[end-1] = nodes[end-1], nodes[idx]
    bd.left_len -= 1
    return nodes[end-1]

def solve_second_graph(nodes: List[int], bd: Bidomain, wselected: List[bool]) -> int:
    idx = selectW_index(nodes, rewards, bd.r, bd.right_len, wselected)
    bd.right_len -= 1
    wselected[nodes[idx]] = True
    return nodes[idx]



def calc_bound(bidomains: List[Bidomain]) -> int:
    bound = 0
    for bidomain in bidomains:
        bound += min(bidomain.left_len, bidomain.right_len)
    return bound


def select_bidomain(domains: List[Bidomain], left: List[int], rewards:DoubleQRewards, current_matching_size: int):
    # Select the bidomain with the smallest max(leftsize, rightsize), breaking
    # ties on the smallest vertex index in the left set
    min_size = np.inf
    min_tie_breaker = np.inf
    best = -1
    for i in range(len(domains)):
        bd = domains[i]
        if current_matching_size > 0 and not bd.is_adjacent:
            continue
        len_bd = max(bd.left_len, bd.right_len)
        if len_bd < min_size:
            min_size = len_bd
            min_tie_breaker = left[bd.l+selectV_index(left, rewards, bd.l, bd.left_len)]
            best = i
        elif len_bd == min_size:
            tie_breaker = left[bd.l+selectV_index(left, rewards, bd.l, bd.left_len)]
            if tie_breaker < min_tie_breaker:
                min_tie_breaker = tie_breaker
                best = i
    return best


def selectV_index(arr: List[int], rewards:DoubleQRewards, start_idx: int, length: int):
    idx = -1
    max_g = -1
    best_vtx = np.inf
    for i in range(length):
        vtx = arr[start_idx + i]
        vtx_reward = rewards.get_vertex_reward(vtx, False)
        if vtx_reward > max_g:
            idx = i
            best_vtx = vtx
            max_g = vtx_reward
        elif vtx_reward == max_g:
            if vtx < best_vtx:
                idx = i
                best_vtx = vtx
    return idx


def selectW_index(arr:List[int], rewards:DoubleQRewards, v:int, start_idx: int, length: int, wselected:List[int]):
    idx = -1
    max_g = -1
    best_vtx = float('inf')

    for i in range(length):
        vtx = arr[start_idx + i]
        if wselected[vtx] == 0:
            pair_reward = rewards.get_pair_reward(v, vtx, False)

            # Check if this is the best pair so far
            if pair_reward > max_g:
                idx = i
                best_vtx = vtx
                max_g = pair_reward
            elif pair_reward == max_g:
                if vtx < best_vtx:
                    idx = i
                    best_vtx = vtx

    return idx


def filter_domains():
    pass


def abort_due_to_timeout(start) -> bool:
    return 0 < opt.timeout < time.monotonic() - start
