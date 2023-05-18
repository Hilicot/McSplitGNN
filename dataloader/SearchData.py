from __future__ import annotations
import numpy as np
import struct


class SearchData:
    is_valid: bool
    left_bidomain: np.ndarray[int]
    vertex_scores: np.ndarray[int]
    g0: Graph
    g1: Graph

    def __init__(self, f):
        self.is_valid = self.read_binary_data(f) >= 0

    def __str__(self):
        return f"SearchData: {self.left_bidomain}, {self.vertex_scores}"

    def read_binary_data(self, f) -> int:
        n_bytes = f.read(4)
        if not n_bytes:
            return -1
        n = struct.unpack("i", n_bytes)[0]
        m_bytes = f.read(4)
        m = struct.unpack("i", m_bytes)[0]
        left_bidomain_bytes = f.read(n*4)
        vertex_scores_bytes = f.read(m*4)
        _left_bidomain = struct.unpack(f"{n}i", left_bidomain_bytes)
        _vertex_scores = struct.unpack(f"{m}i", vertex_scores_bytes)
        self.left_bidomain = np.array(_left_bidomain, dtype=np.int)
        self.vertex_scores = np.array(_vertex_scores, dtype=np.int)
        return m

    def set_graphs(self, graphs:Tuple[Graph,Graph]):
        # TODO instead of saving the entire graphs, use the bidomains to already create the bidomain graphs (as nxgraphs?)
        self.g0 = graphs[0]
        self.g1 = graphs[1]

class SearchDataW(SearchData):
    v: int
    right_bidomain: np.ndarray[int]
    bounds: np.ndarray[int]

    def __str__(self):
        return f"SearchDataW: {self.left_bidomain}, {self.vertex_scores}, {self.v}, {self.right_bidomain}, {self.bounds}"

    def read_binary_data(self, f) -> int:
        m = super().read_binary_data(f)
        if m < 0:
            return m
        v_bytes = f.read(4)
        self.v = struct.unpack("i", v_bytes)[0]
        right_bidomain_bytes = f.read(m*4)
        bounds_bytes = f.read(m*4)
        _right_bidomain = struct.unpack(f"{m}i", right_bidomain_bytes)
        _bounds = struct.unpack(f"{m}i", bounds_bytes)
        self.right_bidomain = np.array(_right_bidomain, dtype=np.int)
        self.bounds = np.array(_bounds, dtype=np.int)
        return m
