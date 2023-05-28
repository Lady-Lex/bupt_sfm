import networkx as nx
import matplotlib.pyplot as plt  # 导入模块函数
from typing import Any, List, Tuple, Dict, Optional

from .feature import *


class CoFeaturesNode:
    def __init__(self, node_id, image, intrinsics):
        self.node_id = node_id
        self.image = image
        self.intrinsics = intrinsics
        self.pose = None


class CoFeaturesEdge:
    def __init__(self, node_id1, node_id2, feature1, feature2, weight, E):
        self.node_id1 = node_id1
        self.node_id2 = node_id2
        self.weight = weight
        self.feature1 = feature1
        self.feature2 = feature2
        self.EssentialMatrix = E


class CoFeaturesGraph(nx.Graph):
    def __init__(self):
        super(CoFeaturesGraph, self).__init__()
        self._nodes = {}
        self._edges = {}
        self.max_weight = 0
        self.max_weight_edge = None

    def reset(self):
        self._nodes = {}
        self._edges = {}
        self.max_weight = 0
        self.max_weight_edge = None
        self.clear()

    def add_node(self, node_id, image, intrinsics):
        self._nodes[node_id] = CoFeaturesNode(node_id, image, intrinsics)
        super(CoFeaturesGraph, self).add_node(node_id)

    def add_edge(self, node_id1, node_id2, feature1, feature2, weight, E):
        if node_id1 >= node_id2:
            raise ValueError("node_id1 should be smaller than node_id2")
        self._edges[(node_id1, node_id2)] = CoFeaturesEdge(node_id1, node_id2, feature1, feature2, weight, E)
        super(CoFeaturesGraph, self).add_edge(node_id1, node_id2, weight=weight)
        if weight > self.max_weight:
            self.max_weight = weight
            self.max_weight_edge = self._edges[(node_id1, node_id2)]

    def get_node(self, node_id):
        return self._nodes[node_id]

    def get_edge(self, node_id1, node_id2):
        if node_id1 > node_id2:
            temp_node_id1 = node_id2
            temp_node_id2 = node_id1
        elif node_id1 < node_id2:
            temp_node_id1 = node_id1
            temp_node_id2 = node_id2
        else:
            raise ValueError("node_id1 should be different from node_id2")
        return self._edges[(temp_node_id1, temp_node_id2)]

    def get_nodes(self):
        return self._nodes

    def get_edges(self):
        return self._edges

    def get_max_weight_edge(self):
        return self.max_weight_edge, self.max_weight

    def all_neighbors(self, node_ids: int, sort=True) -> Tuple[List[int], List[Tuple[int, int]]]:
        neighbor_nodes = set()
        neighbor_edges = set()
        for node_id in node_ids:
            for neighbor_id in self.neighbors(node_id):
                neighbor_nodes.add(neighbor_id)
                neighbor_edges.add(self.get_edge(node_id, neighbor_id))
        
        neighbor_edges = list(neighbor_edges)
        if sort:
            neighbor_edges = sorted(neighbor_edges, key=lambda x: x.weight)

        return list(neighbor_nodes), neighbor_edges

    def del_edge(self, node_id1, node_id2):
        self._edges.pop((node_id1, node_id2))
        self.max_weight = -1
        self.max_weight_edge = None
        super(CoFeaturesGraph, self).remove_edge(node_id1, node_id2)

    def draw(self):
        nx.draw(self, with_labels=True)
        plt.show()


# class CovisibilityGraph(nx.Graph):
#     def __init__(self):
#         super(CovisibilityGraph, self).__init__()
#         self._nodes = {}
#         self._edges = {}
#
#     def reset(self):
#         self._nodes = {}
#         self._edges = {}
#         self.clear()
#
#     def add_node(self, node_id, node):
#         self._nodes[node_id] = node
#         super(CovisibilityGraph, self).add_node(node_id)
#
#     def add_edge(self, node_id1, node_id2, weight):
#         self._edges[(node_id1, node_id2)] = weight
#         super(CovisibilityGraph, self).add_edge(node_id1, node_id2, weight=weight)
#
#     def get_node(self, node_id):
#         return self._nodes[node_id]
#
#     def get_edge(self, node_id1, node_id2):
#         return self._edges[(node_id1, node_id2)]
#
#     def get_nodes(self):
#         return self._nodes
#
#     def get_edges(self):
#         return self._edges
#
#     def draw(self):
#         nx.draw(self)
#         plt.show()


_co_features_graph = CoFeaturesGraph()
# _covisibility_graph = CovisibilityGraph()
