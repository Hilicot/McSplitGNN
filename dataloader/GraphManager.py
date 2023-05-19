from __future__ import annotations
from src.graph import Graph, readAsciiGraph
import os
import struct
import numpy as np

class GraphManager:
    graph_pairs: Dict[str, Tuple[Graph, Graph]]

    def __init__(self):
        self.graph_pairs:Dict[GraphPair] = {}

    def read_graph_pair(self, folder) -> GraphPair:
        if folder in self.graph_pairs:
            return self.graph_pairs[folder]
        g0_path = os.path.join(folder, "g0")
        g1_path = os.path.join(folder, "g1")
        g0 = readAsciiGraph(g0_path)
        g1 = readAsciiGraph(g1_path)
        solution = read_solution(folder)
        graph_pair = GraphPair(g0, g1, solution)
        self.graph_pairs[folder] = graph_pair
        return graph_pair

    def get_graph_pair(self) -> Tuple[Graph, Graph]:
        return self.graph_pairs[folder]

def read_solution(folder) -> np.array:
    path = os.path.join(folder, "solution")
    solution = []
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(4), byteorder='little')
        for i in range(size):
            v = int.from_bytes(f.read(4), byteorder='little')
            w = int.from_bytes(f.read(4), byteorder='little')
            solution.append([v, w])
    return np.array(solution)


class GraphPair:
    g0:Graph
    g1: Graph
    solution: np.array

    def __init__(self, g0: Graph, g1: Graph, solution: np.array):
        self.g0 = g0
        self.g1 = g1
        self.solution = solution

    def __str__(self):
        return "g0:\n" + str(self.g0) + "\ng1:\n" + str(self.g1) + "\nsolution:\n" + str(self.solution)

    def __eq__(self, other):
        return self.g0 == other.g0 and self.g1 == other.g1 and self.solution == other.solution

    def __ne__(self, other):
        return not self.__eq__(other)