from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from src.graph import Graph
from model import bidomain_to_gnn_data
from typing import Union, List


class ModelGNN(nn.Module):
    def __init__(self):
        super(ModelGNN, self).__init__()

    def get_prediction(self, g: Graph, bidomain: Union[np.ndarray[int], List]) -> int:
        self.eval()
        data, _ = bidomain_to_gnn_data(g, bidomain)

        # if the graph is disconnected, the GNN cannot infer anything, so select the first vertex (we assume it's the one with the best heuristic)
        if len(data.edge_index) == 0:
            return 0

        with torch.no_grad():
            prediction = self.forward(data)
        return torch.argmax(prediction, dim=0).item()
