from __future__ import annotations
from src.graph import Graph, readAsciiGraph
import os


class GraphManager:
    graph_pairs: Dict[str, Tuple[Graph, Graph]]

    def __init__(self):
        self.graph_pairs = {}

    def read_graph_pair(self, folder) -> Tuple[Graph, Graph]:
        if folder in self.graph_pairs:
            return self.graph_pairs[folder]
        g0_path = os.path.join(folder, "g0")
        g1_path = os.path.join(folder, "g1")
        g0 = readAsciiGraph(g0_path)
        g1 = readAsciiGraph(g1_path)
        self.graph_pairs[folder] = (g0, g1)
        return g0, g1

    def get_graph_pair(self) -> Tuple[Graph, Graph]:
        return self.graph_pairs[folder]