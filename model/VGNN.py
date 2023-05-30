from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from .ModelGNN import ModelGNN

class VGNN(ModelGNN):
    def __init__(self):
        super(VGNN, self).__init__()
        self.gc1 = GCNConv(1, 4)
        self.gc2 = GCNConv(4, 16)
        self.gc3 = GCNConv(16, 8)
        self.nn1 = nn.Linear(8, 4)
        self.nn2 = nn.Linear(4, 1)
        self.relu = nn.ReLU()

        # init weights
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.xavier_uniform_(self.nn2.weight)

    def forward(self, data):
        x, adj = data.x, data.edge_index
        x = self.gc1(x, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        x = self.relu(x)
        x = self.gc3(x, adj)
        x = self.relu(x)
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        return torch.sigmoid(x)
