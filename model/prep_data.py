from __future__ import annotations
import numpy as np
from src.graph import Graph
from torch_geometric.data import Data
import torch
from typing import Dict, List, Tuple, Union

def bidomain_to_gnn_data(g: Graph, bidomain: Union[np.ndarray[int], List[int]]) -> Tuple[Data, Dict[int, int]]:
    """Convert bidomain to a graph in Data format. Docs: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs"""
    edges = []
    mapping = {}
    for i, vtx in enumerate(bidomain):
        mapping[vtx] = i

    for i, vtx in enumerate(bidomain):
        for j in g.adjlist[vtx].adjNodes:
            if j.id in bidomain:
                edges.append([i, mapping[j.id]])

    # adjacency matrix: sparse tensor in COO format. each edge is represented twice (both directions)
    adjacency_matrix = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # node features: each node has 1 single feature equal to 1 (we only care about the size of the graph)
    node_features = torch.ones(len(bidomain), dtype=torch.float32).unsqueeze(1)  # add channel dim
    return Data(x=node_features, edge_index=adjacency_matrix), mapping
