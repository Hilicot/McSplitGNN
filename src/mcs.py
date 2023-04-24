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

    logging.debug(f"Found {len(domains)} starting domains.")

    g0_matched = [False]*g0.n
    g1_matched = [False]*g1.n
    incumbent = solve(g0, g1, rewards, g0_matched, g1_matched, domains, left, right, 1)

    return incumbent

class Step:
    def __init__(self, domains, wselected, w_iter=-1, v=-1, cur_len=0):
        self.domains = domains
        self.wselected = wselected
        self.w_iter = w_iter
        self.bd = None
        self.v = v
        self.cur_len = cur_len
        self.bd_idx = -1


def solve(g0, g1, rewards, g0_matched, g1_matched, domains, left, right, matching_size_goal):
    nodes = 0
    steps = [Step(domains, set())]
    incumbent = []
    current = []
    start = time.monotonic()

    while len(steps) > 0:
        s = steps[-1]

        # check timeout
        if abort_due_to_timeout(start):
            logging.info("Timeout")
            return incumbent

        # delete eventual extra vertices from previous iterations
        while len(current) > s.cur_len:
            pr = current.pop()
            g0_matched[pr.v] = 0
            g1_matched[pr.w] = 0

        """ V-step """
        if s.w_iter == -1:
            nodes += 1

            # check max iterations
            if 0 < opt.max_iter < nodes:
                logging.info(f"Reached {nodes} iterations")
                return incumbent

            # If the current matching is larger than the incumbent matching, update the incumbent
            if len(current) > len(incumbent):
                incumbent.clear()
                incumbent.extend(current)
                if not opt.quiet:
                    print(f"Incumbent size: {len(incumbent)}")

                rewards.update_policy_counter(True)

            # Prune the branch if the upper bound is too small
            bound = len(current) + calc_bound(s.domains)
            if bound <= len(incumbent) or bound < matching_size_goal:
                steps.pop()
                # print(f"nodes: {nodes} pruned")
                continue

            # Select a bidomain based on the heuristic
            bd_idx = select_bidomain(s.domains, left, rewards, len(current))
            if bd_idx == -1:
                # In the MCCS case, there may be nothing we can branch on
                continue
            bd = s.domains[bd_idx]

            # Select vertex v (vertex with max reward)
            tmp_idx = selectV_index(left, rewards, bd.l, bd.left_len)
            v = left[bd.l + tmp_idx]
            bd.left_len -= 1
            left[bd.l + tmp_idx], left[bd.l + bd.left_len] = left[bd.l + bd.left_len], left[bd.l + tmp_idx]
            rewards.update_policy_counter(False)

            # Next iteration try to select a vertex w to pair with v
            s2 = Step(s.domains, set(), 0, v, cur_len=len(current))
            s2.bd = bd
            s2.bd_idx = bd_idx
            steps.append(s2)
            s2.bd.right_len -= 1
            continue

        """ W-step """
        if s.w_iter < s.bd.right_len + 1:
            tmp_idx = selectW_index(right, rewards, s.v, s.bd.r, s.bd.right_len + 1, s.wselected)
            w = right[s.bd.r + tmp_idx]
            s.wselected.add(w)
            right[s.bd.r + tmp_idx], right[s.bd.r + s.bd.right_len] = right[s.bd.r + s.bd.right_len], right[
                s.bd.r + tmp_idx]
            rewards.update_policy_counter(False)

            if nodes%100000 == 0:
                logging.debug(f"nodes: {nodes}, v: {s.v}, w: {w}, size: {len(current)}, dom: {s.bd.left_len} {s.bd.right_len}")

            cur_len = len(current)
            result = generate_new_domains(s.domains, current, g0_matched, g1_matched, left, right, g0, g1, s.v, w)
            rewards.update_rewards(result, s.v, w)

            s.w_iter += 1
            s.cur_len = cur_len

            # next iterations select a new vertex v
            steps.append(Step(result[0], s.wselected, -1, cur_len=len(current)))
            continue

        """ Backtrack """
        # If we have tried all vertices w, we are done with this vertex v and this bidomain
        s.bd.right_len += 1
        if s.bd.left_len == 0:
            s.domains[s.bd_idx] = s.domains[-1]
            s.domains.pop()
        s = steps.pop()
        steps[-1].cur_len = s.cur_len

    return incumbent


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
        wselected: Set[int],
):
    idx = -1
    max_g = -1
    best_vtx = float("inf")

    for i in range(length):
        vtx = arr[start_idx + i]
        if vtx not in wselected:
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


def generate_new_domains(
        bidomains: List[Bidomain],
        current_sol: List[VertexPair],
        g0_matched: List[bool],
        g1_matched: List[bool],
        left: List[int],
        right: List[int],
        g0: Graph,
        g1: Graph,
        v: int,
        w: int
) -> Tuple[List[Bidomain], int]:
    current_sol.append(VertexPair(v, w))
    g0_matched[v] = True
    g1_matched[w] = True
    new_d: List[Bidomain] = []
    temp, total = 0, 0

    leaves_match_size = 0
    i = j = 0

    while i < len(g0.leaves[v]) and j < len(g1.leaves[w]):
        if g0.leaves[v][i].first < g1.leaves[w][j].first:
            i += 1
        elif g0.leaves[v][i].first > g1.leaves[w][j].first:
            j += 1
        else:
            leaf0 = g0.leaves[v][i].second
            leaf1 = g1.leaves[w][j].second
            p = q = 0
            while p < len(leaf0) and q < len(leaf1):
                if g0_matched[leaf0[p]]:
                    p += 1
                elif g1_matched[leaf1[q]]:
                    q += 1
                else:
                    v_leaf = leaf0[p]
                    w_leaf = leaf1[q]
                    p += 1
                    q += 1
                    current_sol.append(VertexPair(v_leaf, w_leaf))
                    g0_matched[v_leaf] = True
                    g1_matched[w_leaf] = True
                    leaves_match_size += 1
            i += 1
            j += 1

    for old_bd in bidomains:
        l = old_bd.l
        r = old_bd.r

        if leaves_match_size > 0 and not old_bd.is_adjacent:
            unmatched_left_len = remove_matched_vertex(left, l, old_bd.left_len, g0_matched)
            unmatched_right_len = remove_matched_vertex(right, r, old_bd.right_len, g1_matched)
        else:
            unmatched_left_len = old_bd.left_len
            unmatched_right_len = old_bd.right_len

        left_len = partition(left, l, unmatched_left_len, g0, v)
        right_len = partition(right, r, unmatched_right_len, g1, w)
        left_len_noedge = unmatched_left_len - left_len
        right_len_noedge = unmatched_right_len - right_len

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


def remove_matched_vertex(arr: List[int], start: int, len: int, matched: List[bool]) -> int:
    p = 0

    for i in range(len):
        if not matched[arr[start + i]]:
            arr[start + i], arr[start + p] = arr[start + p], arr[start + i]
            p += 1

    return p


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
