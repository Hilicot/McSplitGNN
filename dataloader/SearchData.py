from __future__ import annotations
import numpy as np
import struct
from torch_geometric.data import Data
import torch
from dataloader.GraphManager import GraphPair
from typing import Dict, List, Tuple
from options import opt
from src.graph import Graph

class SearchData:
    is_valid: bool
    graph_pair: GraphPair
    v_vertex_mapping: Dict[int, int] = {}
    data: Data
    skip = False

    def __init__(self, f, graph_pair: GraphPair):
        count, binary_data = self.read_binary_data(f)
        self.is_valid = count > 0
        if self.is_valid:
            self.graph_pair = graph_pair
            self.convert_to_gnn_data(binary_data)

    def __str__(self):
        return f"SearchData: {self.data}"

    def read_binary_data(self, f) -> Tuple[int, Tuple[np.ndarray, np.ndarray]]:
        n_bytes = f.read(4)
        if not n_bytes:
            return -1, ()
        n = struct.unpack("i", n_bytes)[0]
        m_bytes = f.read(4)
        m = struct.unpack("i", m_bytes)[0]
        left_bidomain_bytes = f.read(n*4)
        vertex_scores_bytes = f.read(m*4)
        _left_bidomain = struct.unpack(f"{n}i", left_bidomain_bytes)
        _vertex_scores = struct.unpack(f"{m}i", vertex_scores_bytes)
        return n, (np.array(_left_bidomain, dtype=int), np.array(_vertex_scores, dtype=int))

    def convert_to_gnn_data(self, binary_data) -> Data:
        left_bidomain, vertex_scores = binary_data
        self.data = self.bidomain_to_gnn_data(self.graph_pair.g0, left_bidomain, self.v_vertex_mapping, [pair[0] for pair in self.graph_pair.solution], vertex_scores)
        self.skip = len(self.data.y) == 0 or len(self.data.edge_index) == 0  # don't include bidomains with disconnected vertices (GNN cannot infer anything)
        return self.data.to(opt.device)

    @staticmethod
    def bidomain_to_gnn_data(g: Graph, bidomain: np.ndarray[int], mapping: Dict[int, int], solution_vertices: List, vertex_scores: List) -> Data:
        """Convert bidomain to a graph in Data format. Docs: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs"""
        edges = []
        for i, vtx in enumerate(bidomain):
            mapping[vtx] = i

        for i, vtx in enumerate(bidomain):
            for j in g.adjlist[vtx].adjNodes:
                if j.id in bidomain:
                    edges.append([i, mapping[j.id]])

        # adjacency matrix: sparse tensor in COO format. each edge is represented twice (both directions)
        adjacency_matrix = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # node features: each node has 1 single feature equal to 1 (we only care about the size of the graph)
        node_features = torch.ones(len(bidomain), dtype=torch.float32).unsqueeze(1) # add channel dim
        # labels:
        labels = torch.tensor([abs(vertex_scores[mapping[vtx]]) if vtx in solution_vertices else 0 for vtx in bidomain], dtype=torch.float).t().contiguous().unsqueeze(1)
        labels = torch.functional.F.softmax(labels, dim=0)
        return Data(x=node_features, edge_index=adjacency_matrix, y=labels)


class SearchDataW(SearchData):
    v: int
    right_bidomain: np.ndarray[int]
    bounds: np.ndarray[int]

    def __str__(self):
        return f"SearchDataW: {self.left_bidomain}, {self.vertex_scores}, {self.v}, {self.right_bidomain}, {self.bounds}"

    def read_binary_data(self, f) -> Tuple[int, Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]]:
        count, (left_bidomain, vertex_scores) = super().read_binary_data(f)
        m = len(vertex_scores)
        if count < 0 or m <= 0:
            return -1, ()
        v_bytes = f.read(4)
        v = struct.unpack("i", v_bytes)[0]
        right_bidomain_bytes = f.read(m*4)
        bounds_bytes = f.read(m*4)
        _right_bidomain = struct.unpack(f"{m}i", right_bidomain_bytes)
        _bounds = struct.unpack(f"{m}i", bounds_bytes)
        right_bidomain = np.array(_right_bidomain, dtype=np.int)
        bounds = np.array(_bounds, dtype=np.int)
        return m, (left_bidomain, vertex_scores, v, right_bidomain, bounds)

    def convert_to_gnn_data(self, binary_data) -> Data:
        raise NotImplementedError("SearchDataW does not support convert_to_gnn_data yet")
