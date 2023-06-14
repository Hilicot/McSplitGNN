from __future__ import annotations
import numpy as np
import struct
from torch_geometric.data import Data
import torch
from dataloader.GraphManager import GraphPair
from typing import Dict, List, Tuple
from options import opt
from src.graph import Graph
from model.prep_data import bidomain_to_gnn_data

class SearchData:
    is_valid: bool
    graph_pair: GraphPair
    v_vertex_mapping: Dict[int, int] = {}
    v_data: Data
    labels: torch.Tensor
    scores: torch.Tensor
    heuristics: torch.Tensor
    skip = False

    def __init__(self, f, graph_pair: GraphPair):
        count, binary_data = self.read_binary_data(f)
        self.is_valid = count > 0
        if self.is_valid:
            self.graph_pair = graph_pair
            self.skip = self.convert_to_gnn_data(binary_data)
        # Clean up to make cache smaller
        self.graph_pair = None

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

    def convert_to_gnn_data(self, binary_data) -> bool:
        left_bidomain, vertex_scores = binary_data
        self.v_data, self.v_vertex_mapping = bidomain_to_gnn_data(self.graph_pair.g0, left_bidomain)
        self.v_data = self.v_data.to(opt.device)

        self.heuristics = torch.tensor(self.graph_pair.g0_heuristic[left_bidomain], dtype=torch.float)
        self.create_labels(self.v_vertex_mapping, [pair[0] for pair in self.graph_pair.solution], vertex_scores)
        self.v_vertex_mapping = None
        return len(self.labels) == 0 or len(self.v_data.edge_index) == 0  # don't include bidomains with disconnected vertices (GNN cannot infer anything)


    def create_labels(self, mapping, solution_vertices: List, vertex_scores=None):
        if vertex_scores is None:
            vertex_scores = np.ones(len(mapping.keys()), dtype=int)
        self.labels = torch.tensor([1 if vtx in solution_vertices else 0 for vtx, i in mapping.items()],
                              dtype=torch.float)
        self.scores = torch.tensor(vertex_scores, dtype=torch.float)


class SearchDataW(SearchData):
    w_data: Data
    w_vertex_mapping: Dict[int, int] = {}

    def __init__(self, f, graph_pair: GraphPair):
        super().__init__(f, graph_pair)

        # Clean up to make cache smaller
        self.v_data = None # remove it since we don't use it

    def __str__(self):
        return f"SearchDataW"

    def read_binary_data(self, f) -> Tuple[int, Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]]:
        count, binary_data = super().read_binary_data(f)
        if count < 0:
            return -1, ()
        left_bidomain, vertex_scores = binary_data
        m = len(vertex_scores)
        if m <= 0:
            return -1, ()
        v_bytes = f.read(4)
        v = struct.unpack("i", v_bytes)[0]
        right_bidomain_bytes = f.read(m*4)
        bounds_bytes = f.read(m*4)
        _right_bidomain = struct.unpack(f"{m}i", right_bidomain_bytes)
        _bounds = struct.unpack(f"{m}i", bounds_bytes)
        right_bidomain = np.array(_right_bidomain, dtype=int)
        bounds = np.array(_bounds, dtype=int)
        return m, (left_bidomain, vertex_scores, v, right_bidomain, bounds)

    def convert_to_gnn_data(self, binary_data) -> bool:
        left_bidomain, vertex_scores, v, right_bidomain, bounds = binary_data

        # bidomains
        #self.v_data, self.v_vertex_mapping = self.bidomain_to_gnn_data(self.graph_pair.g0, left_bidomain)
        #self.v_data = self.v_data.to(opt.device)
        self.w_data, self.w_vertex_mapping = bidomain_to_gnn_data(self.graph_pair.g1, right_bidomain)
        self.w_data = self.w_data.to(opt.device)

        # labels
        self.heuristics = torch.tensor(self.graph_pair.g1_heuristic[right_bidomain], dtype=torch.float)
        self.create_labels(self.w_vertex_mapping, [pair[1] for pair in self.graph_pair.solution], vertex_scores)
        self.w_vertex_mapping = None

        return len(self.labels) == 0 or len(self.w_data.edge_index) == 0
