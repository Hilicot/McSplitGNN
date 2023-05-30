import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
from .ModelGNN import ModelGNN

class WGNN(ModelGNN):
    def __init__(self):
        super(WGNN, self).__init__()
        self.gc1v = GCNConv(1, 4)
        self.gc2v = GCNConv(4, 8)
        self.gc1w = GCNConv(1, 4)
        self.gc2w = GCNConv(4, 8)
        self.relu = nn.ReLU()
        self.nn1 = nn.Linear(8, 4)
        self.nn2 = nn.Linear(4, 1)

    def forward(self, data:Data):
        v_data, w_data, _ = data
        xv, v_domain = v_data.x, v_data.edge_index
        xw, w_domain = w_data.x, w_data.edge_index
        xv = self.gc1v(xv, v_domain)
        xv = self.relu(xv)
        xv = self.gc2v(xv, v_domain)
        xw = self.gc1w(xw, w_domain)
        xw = self.relu(xw)
        xw = self.gc2w(xw, w_domain)
        x = torch.cat((xv, xw), dim=0)
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        return F.log_softmax(x, dim=1)