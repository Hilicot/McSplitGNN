from abc import ABC, abstractmethod
import numpy as np
from model import bidomain_to_gnn_data
from src.graph import Graph
import logging
from model.VGNN import VGNN
import torch
import os
from options import opt


class SortBase(ABC):
    @abstractmethod
    def __name__(self):
        return "Base"

    @abstractmethod
    def sort(self, g: Graph):
        raise NotImplementedError


class SortDegree(SortBase):
    def __name__(self):
        return "Degree"

    def sort(self, g: Graph):
        return [len(g.adjlist[i].adjNodes) for i in range(g.n)]


class SortPagerank(SortBase):
    def __name__(self):
        return "Pagerank"

    def sort(self, g: Graph):
        damping_factor = 0.85
        epsilon = 0.00001
        out_links = np.zeros(g.n)
        for i in range(g.n):
            out_links[i] = len(g.adjlist[i].adjNodes)
        stochastic_g = np.zeros([g.n, g.n])
        for i in range(g.n):
            if not out_links[i]:
                stochastic_g[i][:] = 1.0/g.n
            else:
                for w in g.adjlist[i].adjNodes:
                    stochastic_g[i][w.id] = 1.0/out_links[i]
        p = np.ones(g.n)/g.n
        transposed = stochastic_g.transpose()
        while True:
            ranks = np.zeros(g.n)
            for i in range(g.n):
                ranks[i] += np.dot(transposed[i], p)
            ranks = damping_factor*ranks + (1.0 - damping_factor)/g.n
            error = np.sum(np.abs(ranks - p))
            if error < epsilon:
                break
            p = ranks

        return (ranks//epsilon).astype(int)


class SortGNN(SortBase):
    def __name__(self):
        return "GNN"

    def sort(self, g: Graph):
        logging.debug("Loading V model")
        v_model = VGNN()
        v_model.load_state_dict(torch.load(os.path.join(opt.model_folder, "WGNN.pt")))
        v_model.eval()

        data, _ = bidomain_to_gnn_data(g, np.arange(g.n))
        scores = np.zeros(g.n)

        # if the graph is disconnected, the GNN cannot infer anything
        if len(data.edge_index) == 0:
            return scores

        with torch.no_grad():
            scores = v_model.forward(data)

        return scores*1000


heuristics = {
    "Degree": SortDegree(),
    "Pagerank": SortPagerank(),
    "GNN": SortGNN()
}
