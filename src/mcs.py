from __future__ import annotations
from typing import List, Tuple
import time
import numpy as np
import logging

from options import opt
from src.vertex_pair import VertexPair
from src.reward import DoubleQRewards
from src.graph import Graph
from src.bidomain import Bidomain


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

    logging.info(f"Found {len(domains)} starting domains.")

    incumbent = solve(g0, g1, rewards, domains, left, right)

    return incumbent


def solve(
    g0: Graph,
    g1: Graph,
    rewards: DoubleQRewards,
    starting_bidomain: List[Bidomain],
    left: List[int],
    right: List[int],
) -> List[VertexPair]:
    best_sol: List[VertexPair] = []
    start = time.monotonic()
    max_size = max(g0.n, g1.n) * 2
    depth = 0
    current_sol: List[VertexPair] = []
    current_bidomain = [0] * max_size
    bidomains: List[List[Bidomain]] = [starting_bidomain]
    wselected = [False] * g1.n
    g0_matched = [0] * g0.n
    g1_matched = [0] * g1.n

    v = np.inf
    w = -1

    nodes = 0
    iteration = 1

    while depth >= 0:
        if abort_due_to_timeout(start):
            logging.info("Timeout")
            return best_sol

        if opt.max_iter and iteration >= 2*opt.max_iter:
            logging.info(f"Reached {iteration} iterations and {nodes} v iterations")
            return best_sol

        if depth % 2 == 0:
            nodes += 1
            bound = len(current_sol) + calc_bound(bidomains[depth // 2])

            if bound <= len(best_sol):
                depth -= 1
                if depth < 0:
                    continue
                v, w = current_sol[-1].get()
                current_sol.pop()
                bidomains.pop()
                bidomains[depth // 2][current_bidomain[depth // 2]].right_len += 1
                continue
            
            w = -1
            current_bidomain[depth // 2] = select_bidomain(
                bidomains[depth // 2], left, rewards, len(current_sol)
            )
            if current_bidomain[depth // 2] == np.inf:
                depth -= 1
                v, w = current_sol[-1].get()
                current_sol.pop()
                bidomains.pop()
                bidomains[depth // 2][current_bidomain[depth // 2]].right_len += 1
                continue
            v = solve_first_graph(
                left, bidomains[depth // 2][current_bidomain[depth // 2]], rewards
            )
            # logging.debug(f'Depth={depth} | Selected v={v}, current bidomain {bidomains[depth // 2][current_bidomain[depth // 2]]}')
            wselected = [False] * g1.n
            iteration += 1
            depth += 1
        else:
            w = solve_second_graph(
                right,
                bidomains[depth // 2][current_bidomain[depth // 2]],
                wselected,
                rewards,
                v,
            )
            iteration += 1
            # logging.debug(f'Depth={depth} | Selected w={w}, current bidomain {bidomains[depth // 2][current_bidomain[depth // 2]]}')
            if w != -1:
                current_sol.append(VertexPair(v, w))

                if len(current_sol) > len(best_sol):
                    best_sol = [x for x in current_sol]
                    lap = time.monotonic()
                    time_elapsed = lap - start

                    if not opt.quiet and (nodes - len(best_sol) > 10 or len(best_sol) % 100 == 0):
                        logging.info(f"Incumbent size: {len(best_sol)} \t Iterations: {iteration} \t Time: {time_elapsed}")

                    rewards.update_policy_counter(True)

                new_domains, total = filter_domains(
                    bidomains[depth // 2],
                    left,
                    right,
                    g0,
                    g1,
                    v,
                    w,
                    # opt.directed or opt.edge_labelled because always False for us
                )

                bidomains.append(new_domains)
                rewards.update_rewards((total, len(new_domains)), v, w)

                depth += 1
            else:
                bidomains[depth // 2][current_bidomain[depth // 2]].right_len += 1
                depth -= 1

                if bidomains[depth // 2][current_bidomain[depth // 2]].left_len <= 0:
                    bidomains[depth // 2][-1], bidomains[depth // 2][current_bidomain[depth // 2]] = bidomains[depth // 2][current_bidomain[depth // 2]], bidomains[depth // 2][-1]
                    bidomains[depth // 2] =  bidomains[depth // 2][:-1]

    return best_sol


def solve_first_graph(nodes: List[int], bd: Bidomain, rewards: DoubleQRewards) -> int:
    end = bd.l + bd.left_len
    idx = selectV_index(nodes, rewards, bd.l, bd.left_len)
    # put vertex at the back
    if idx != np.inf:
        nodes[bd.l + idx], nodes[end - 1] = nodes[end - 1], nodes[bd.l + idx]
    bd.left_len -= 1
    rewards.update_policy_counter(False)
    return nodes[end - 1]


def solve_second_graph(
    nodes: List[int],
    bd: Bidomain,
    wselected: List[bool],
    rewards: DoubleQRewards,
    v: int,
) -> int:
    idx = selectW_index(nodes, rewards, v, bd.r, bd.right_len, wselected)
    if idx != np.inf:
        nodes[bd.r + idx], nodes[bd.r + bd.right_len - 1] = nodes[bd.r + bd.right_len - 1], nodes[bd.r + idx]
    bd.right_len -= 1
    wselected[nodes[bd.r + bd.right_len]] = True
    rewards.update_policy_counter(False)
    return nodes[bd.r + bd.right_len]


def calc_bound(bidomains: List[Bidomain]) -> int:
    bound = 0
    for bidomain in bidomains:
        bound += min(bidomain.left_len, bidomain.right_len)
    return bound


def select_bidomain(
    domains: List[Bidomain],
    left: List[int],
    rewards: DoubleQRewards,
    current_matching_size: int,
):
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
            min_tie_breaker = left[
                bd.l + selectV_index(left, rewards, bd.l, bd.left_len)
            ]
            best = i
        elif len_bd == min_size:
            tie_breaker = left[bd.l + selectV_index(left, rewards, bd.l, bd.left_len)]
            if tie_breaker < min_tie_breaker:
                min_tie_breaker = tie_breaker
                best = i
    return best


def selectV_index(arr: List[int], rewards: DoubleQRewards, start_idx: int, length: int):
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


def selectW_index(
    arr: List[int],
    rewards: DoubleQRewards,
    v: int,
    start_idx: int,
    length: int,
    wselected: List[int],
):
    idx = -1
    max_g = -1
    best_vtx = float("inf")

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


def filter_domains(
    bidomains: List[Bidomain],
    left: List[int],
    right: List[int],
    g0: Graph,
    g1: Graph,
    v: int,
    w: int,
) -> Tuple[List[Bidomain], int]:
    # TODO: LUM
    new_d: List[Bidomain] = []
    temp, total = 0, 0

    for old_bd in bidomains:
        l = old_bd.l
        r = old_bd.r
        left_len = partition(left, l, old_bd.left_len, g0, v)
        right_len = partition(right, r, old_bd.right_len, g1, w)
        left_len_noedge = old_bd.left_len - left_len
        right_len_noedge = old_bd.right_len - right_len

        # compute reward
        temp = (
            min(old_bd.left_len, old_bd.right_len)
            - min(left_len, right_len)
            - min(left_len_noedge, right_len_noedge)
        )
        total += temp

        if left_len_noedge and right_len_noedge:
            new_d.append(
                Bidomain(
                    l + left_len,
                    r + right_len,
                    left_len_noedge,
                    right_len_noedge,
                    old_bd.is_adjacent,
                )
            )

        if left_len and right_len:
            new_d.append(Bidomain(l, r, left_len, right_len, True))

    return new_d, total


def partition(all_vv: List[int], start: int, len: int, g: Graph, index: int):
    i = j = 0

    while j < len:
        if g.get(index, all_vv[start + j]):
            all_vv[start + i], all_vv[start + j] = all_vv[start + j], all_vv[start + i]
            i += 1

        j += 1

    return i


def abort_due_to_timeout(start) -> bool:
    return 0 < opt.timeout < time.monotonic() - start
