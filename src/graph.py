from __future__ import annotations
from typing import List
import os
import logging

from options import opt


class Node:
    id: int
    label: int
    adjNodes: List[Node]

    def __init__(self, id, label):
        self.id = id
        self.label = label << 1
        self.adjNodes = []


class Graph:
    n: int
    e: int
    adjlist: List[Node]
    leaves: List[Node]

    def __init__(self, n):
        self.n = n
        self.e = 0
        self.adjlist = [Node(i, 0) for i in range(n)]
        self.leaves = [[] for i in range(n)]

    def get(self, u, v):
        if u < self.n:
            for node in self.adjlist[u].adjNodes:
                if node.id == v:
                    return 1
        return 0   

    def pack_leaves(self):
        deg = [len(self.adjlist[i].adjNodes) for i in range(self.n)]

        for u in range(self.n):
            for v in range(len(self.adjlist[u].adjNodes)):
                if deg[v] == 1:
                    labels = (1, self.adjlist[v].label)
                    pos = -1
                    for k in range(len(self.leaves[u])):
                        if self.leaves[u][k][0] == labels:
                            pos = k
                            break
                        if k == len(self.leaves[u]) - 1:
                            self.leaves[u].append((labels, []))
                    self.leaves[u][pos][1].append(v)

            self.leaves[u].sort()

    def add_edge(self, u: int, v: int):
        self.e += 1
        if u != v:
            self.adjlist[u].adjNodes.append(Node(v, 0))
            self.adjlist[v].adjNodes.append(Node(u, 0))
        else:
            self.adjlist[u].label |= 1

    def computeNumEdges(self):
        self.e = 0
        for i in range(self.n):
            self.e += len(self.adjlist[i].adjNodes)
        self.e //= 2
        return self.e

    def computeDensity(self):
        if self.e == 0:
            self.computeNumEdges()
        return 2 * self.e / (self.n * (self.n - 1))


def induced_subgraph(g: Graph, g_deg) -> Graph:
    # compute mapping vv
    vv = list(range(g.n))
    vv.sort(key=lambda x: g_deg[x], reverse=True)

    # sort graph
    subg = Graph(g.n)
    for i in range(subg.n):
        subg.adjlist[i] = g.adjlist[vv[i]]
        subg.adjlist[i].id = i
        for j in range(len(subg.adjlist[i].adjNodes)):
            subg.adjlist[i].adjNodes[j].id = vv.index(subg.adjlist[i].adjNodes[j].id)

        subg.adjlist[i].adjNodes.sort(key=lambda x: x.id)
    return subg


def read_word(fp):
    a = fp.read(2)
    if len(a) != 2:
        raise Exception("Error reading file.")
    return int.from_bytes(a, byteorder="little", signed=False)


def readBinaryGraph(filename: str) -> Graph:
    with open(os.path.join(opt.data_folder, filename), "rb") as f:
        nvertices = read_word(f)
        logging.debug("Nvertices:", nvertices)
        g = Graph(nvertices)

        # Labelling scheme: see
        # https://github.com/ciaranm/cp2016-max-common-connected-subgraph-paper/blob/master/code/solve_max_common_subgraph.cc
        m = g.n * 33 // 100
        p = 1
        k1 = 0
        k2 = 0
        while p < m and k1 < 16:
            p *= 2
            k1 = k2
            k2 += 1
        for i in range(nvertices):
            len = read_word(f)
            for j in range(len):
                target = read_word(f)
                g.add_edge(i, target)
    return g


def readAsciiGraph(filename: str) -> Graph:
    with open(os.path.join(opt.data_folder, filename), "r") as f:
        header = f.readline().split()
        nvertices, nedges = int(header[0]), int(header[1])

        logging.debug("nvertices: %d", nvertices)
        g = Graph(nvertices)

        for _ in range(nedges):
            line = f.readline()
            v1, v2 = map(int, line.split())
            g.add_edge(v1, v2)

        if g.e != nedges:
            logging.warning(
                f"Number of edges read ({g.e}) is not equal to the number of edges in the file ({nedges})"
            )
    return g


def readGraph(filename: str) -> Graph:
    if opt.dataset_format == "binary":
        g = readBinaryGraph(filename)
    elif opt.dataset_format == "ascii":
        g = readAsciiGraph(filename)
    else:
        raise Exception("Unknown dataset format: " + opt.dataset_format)

    return g
