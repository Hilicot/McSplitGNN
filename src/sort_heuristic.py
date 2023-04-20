from abc import ABC, abstractmethod
import numpy as np

class SortBase(ABC):
    @abstractmethod
    def __name__(self):
        return "Base"

    @abstractmethod
    def sort(self, g):
        raise NotImplementedError


class SortDegree(SortBase):
    def __name__(self):
        return "Degree"

    def sort(self, g):
        return [len(g.adjlist[i].adjNodes) for i in range(g.n)]


class SortPagerank(SortBase):
    def __name__(self):
        return "Pagerank"

    def sort(self, g):
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
                ranks[i] += np.dot(transposed[i],p)
            ranks = damping_factor*ranks + (1.0 - damping_factor)/g.n
            error = np.sum(np.abs(ranks - p))
            if error < epsilon:
                break
            p = ranks

        return ranks/epsilon

