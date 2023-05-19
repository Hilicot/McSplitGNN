from __future__ import annotations
import numpy as np
import struct
from torch_geometric.data import Data
import torch
from dataloader.GraphManager import GraphPair


class SearchData:
    is_valid: bool
    left_bidomain: np.ndarray[int]
    vertex_scores: np.ndarray[int]
    graph_pair: GraphPair
    v_vertex_mapping: Dict[int, int] = {}

    def __init__(self, f, graph_pair: GraphPair):
        self.is_valid = self.read_binary_data(f)[0] > 0
        self.graph_pair = graph_pair

    def __str__(self):
        return f"SearchData: {self.left_bidomain}, {self.vertex_scores}"

    def read_binary_data(self, f) -> (int, int):
        n_bytes = f.read(4)
        if not n_bytes:
            return -1, -1
        n = struct.unpack("i", n_bytes)[0]
        m_bytes = f.read(4)
        m = struct.unpack("i", m_bytes)[0]
        left_bidomain_bytes = f.read(n*4)
        vertex_scores_bytes = f.read(m*4)
        _left_bidomain = struct.unpack(f"{n}i", left_bidomain_bytes)
        _vertex_scores = struct.unpack(f"{m}i", vertex_scores_bytes)
        self.left_bidomain = np.array(_left_bidomain, dtype=np.int)
        self.vertex_scores = np.array(_vertex_scores, dtype=np.int)
        return n, m

    def convert_to_gnn_data(self) -> Data:
        return self.bidomain_to_gnn_data(self.graph_pair.g0, self.left_bidomain, self.v_vertex_mapping)

    @staticmethod
    def bidomain_to_gnn_data(g: Graph, bidomain: np.ndarray[int], mapping: Dict[int, int]) -> Data:
        """Convert bidomain to a graph in Data format. Docs: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs"""
        edges = []
        mapping = {vtx: i for i, vtx in enumerate(bidomain)}

        for i, vtx in enumerate(bidomain):
            for j in g.adjlist[vtx].adjNodes:
                if j.id in bidomain:
                    edges.append([i, mapping[j.id]])

        # adjacency matrix: sparse tensor in COO format. each edge is represented twice (both directions)
        adjacency_matrix = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # node features: each node has 1 single feature equal to 1 (we only care about the size of the graph)
        node_features = torch.ones(len(bidomain), dtype=torch.float32)
        # labels:
        labels = None  # TODO define labels (attenzione che i vertici sono mappati su label diverse)

        return Data(x=node_features, edge_index=adjacency_matrix, y=labels)


class SearchDataW(SearchData):
    v: int
    right_bidomain: np.ndarray[int]
    bounds: np.ndarray[int]

    def __str__(self):
        return f"SearchDataW: {self.left_bidomain}, {self.vertex_scores}, {self.v}, {self.right_bidomain}, {self.bounds}"

    def read_binary_data(self, f) -> (int, int):
        n, m = super().read_binary_data(f)
        if m < 0:
            return m, m
        v_bytes = f.read(4)
        self.v = struct.unpack("i", v_bytes)[0]
        right_bidomain_bytes = f.read(m*4)
        bounds_bytes = f.read(m*4)
        _right_bidomain = struct.unpack(f"{m}i", right_bidomain_bytes)
        _bounds = struct.unpack(f"{m}i", bounds_bytes)
        self.right_bidomain = np.array(_right_bidomain, dtype=np.int)
        self.bounds = np.array(_bounds, dtype=np.int)
        return m, m

    def convert_to_gnn_data(self) -> Data:
        raise NotImplementedError("SearchDataW does not support convert_to_gnn_data yet")
