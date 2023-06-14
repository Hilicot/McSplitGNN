from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from .ModelGNN import ModelGNN

class DiffGNN(ModelGNN):
    def __init__(self):
        super(DiffGNN, self).__init__()
        self.gc1 = GCNConv(1, 4)
        self.gc2 = GCNConv(4, 16)
        self.gc3 = GCNConv(16, 32)
        self.gc4 = GCNConv(32, 64)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, adj = data.x, data.edge_index
        x = self.gc1(x, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        x = self.relu(x)
        x = self.gc3(x, adj)
        x = self.relu(x)
        x = self.gc4(x, adj)
        return torch.sigmoid(x)
