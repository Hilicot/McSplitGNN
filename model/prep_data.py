from __future__ import annotations
import numpy as np
from src.graph import Graph
from torch_geometric.data import Data
import torch
from typing import Dict, List, Tuple, Union

def bidomain_to_gnn_data(g: Graph, bidomain: Union[np.ndarray[int], List[int]]) -> Tuple[Data, Dict[int, int]]:
    """Convert bidomain to a graph in Data format. Docs: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs"""
    bidomain = np.array(bidomain)
    num_nodes = len(bidomain)
    mapping = np.zeros(np.max(bidomain) + 1, dtype=np.int64)
    mapping[bidomain] = np.arange(num_nodes)

    bidomain_set = set(bidomain)
    edges = []
    for i, vtx in enumerate(bidomain):
        adj_nodes = [adj_node.id for adj_node in g.adjlist[vtx].adjNodes if adj_node.id in bidomain_set]
        edges.extend([[i, mapping[j_id]] for j_id in adj_nodes])
        
    del bidomain_set

    edges = np.array(edges, dtype=np.int64).T
    adjacency_matrix = torch.from_numpy(edges).contiguous()
    node_features = torch.ones(num_nodes, dtype=torch.float32).unsqueeze(1)  # add channel dim

    return Data(x=node_features, edge_index=adjacency_matrix), mapping