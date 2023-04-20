from __future__ import annotations
from vertex_pair import VertexPair
from graph import Graph

def mcs(g0:Graph, g1:Graph) -> List[VertexPair]:
  left: List[int] = []  # the buffer of vertex indices for the left partitions
  right: List[int] = [] # the buffer of vertex indices for the right partitions
  domains = []

  left_labels = set()
  right_labels = set()
  for n in g0.adjlist:
    left_labels.add(n.label)
  for n in g1.adjlist:
    right_labels.add(n.label)
  labels = left_labels.intersection(right_labels)  # labels that appear in both graphs

  # Create a bidomain for each label that appears in both graphs
  for label in labels:
    start_l = len(left)
    start_r = len(right)

    for i in range(g0.n):
      if g0.adjlist[i].label == label:
        left.append(i)
    for i in range(g1.n):
      if g1.adjlist[i].label == label:
        right.append(i)

    left_len = len(left) - start_l
    right_len = len(right) - start_r
    domains.append((start_l, start_r, left_len, right_len, False))

  incumbent = [(-1, -1)]*arguments.prime


  solve(g0, g1, incumbent, domains, left, right)

  test_info.recursions = nodes
  return incumbent



