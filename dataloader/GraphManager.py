from __future__ import annotations
from src.graph import Graph, readAsciiGraph, readBinaryGraph
import os
from typing import Dict, Tuple
import numpy as np
from options import opt
import logging


class GraphManager:
    graph_pairs: Dict[str, GraphPair]

    def __init__(self):
        self.graph_pairs = {}

    def read_graph_pair(self, folder: str) -> GraphPair:
        if folder in self.graph_pairs:
            return self.graph_pairs[folder]
        
        g0_path = os.path.join(folder, "g0")
        g1_path = os.path.join(folder, "g1")
        if opt.dataset_format == "binary":
            g0 = readBinaryGraph(g0_path)
            g1 = readBinaryGraph(g1_path)
        else:
            g0 = readAsciiGraph(g0_path)
            g1 = readAsciiGraph(g1_path)
        solution = read_solution(folder)
        graph_pair = GraphPair(g0, g1, solution)
        self.graph_pairs[folder] = graph_pair

        logging.debug("Graph pair: " + str(folder) + " read")
        return graph_pair

    def get_graph_pair(self, folder: str) -> GraphPair:
        return self.graph_pairs[folder]


def read_solution(folder: str) -> np.array:
    path = os.path.join(folder, "solution")
    solution = []
    
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(4), byteorder="little")
        
        for _ in range(size):
            v = int.from_bytes(f.read(4), byteorder="little")
            w = int.from_bytes(f.read(4), byteorder="little")
            solution.append([v, w])
            
    return np.array(solution)


class GraphPair:
    g0: Graph
    g1: Graph
    solution: np.array
    g0_heuristic: np.array
    g1_heuristic: np.array

    def __init__(self, g0: Graph, g1: Graph, solution: np.array):
        self.g0 = g0
        self.g1 = g1
        self.solution = solution
        logging.debug("Computing heuristics")
        self.g0_heuristic = np.array(opt.sort_heuristic.sort(g0))
        self.g1_heuristic = np.array(opt.sort_heuristic.sort(g1))

    def __str__(self):
        return f'g0:\n{self.g0}\ng1:\n{self.g1}\nsolution:\n{self.solution}'

    def __eq__(self, other):
        return (
            self.g0 == other.g0
            and self.g1 == other.g1
            and self.solution == other.solution
        )

    def __ne__(self, other):
        return not self.__eq__(other)
