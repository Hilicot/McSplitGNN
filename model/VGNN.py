import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class VGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VGNN, self).__init__()

        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, adj = data.x, data.adj
        x = self.gc1(x, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
