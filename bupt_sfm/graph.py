import networkx as nx
import matplotlib.pyplot as plt
from typing import Any, List, Tuple, Dict, Optional

import numpy as np

from .feature import *


class CoFeaturesNode:
    def __init__(self, node_id: int, image: np.ndarray, intrinsics: np.ndarray, all_features: FeaturesData):
        self.node_id = node_id
        self.image = image
        self.intrinsics = intrinsics
        self.all_features = all_features


class CoFeaturesEdge:
    def __init__(self, node_id1: int, node_id2: int, feature1: MatchedFeatures, feature2: MatchedFeatures, weight: int, E: np.ndarray):
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

    def add_node(self, node_id: int, image: np.ndarray, intrinsics: np.ndarray, all_features: FeaturesData):
        self._nodes[node_id] = CoFeaturesNode(node_id, image, intrinsics, all_features)
        super(CoFeaturesGraph, self).add_node(node_id)

    def add_edge(self, node_id1: int, node_id2: int, feature1: MatchedFeatures, feature2: MatchedFeatures, weight: int, E: np.ndarray):
        temp_edge = None
        if node_id1 < node_id2:
            self._edges[(node_id1, node_id2)] = CoFeaturesEdge(node_id1, node_id2, feature1, feature2, weight, E)
            temp_edge = self._edges[(node_id1, node_id2)]
        elif node_id1 > node_id2:
            self._edges[(node_id2, node_id1)] = CoFeaturesEdge(node_id2, node_id1, feature2, feature1, weight, E)
            temp_edge = self._edges[(node_id2, node_id1)]
        else:
            raise ValueError("node_id1 should be different from node_id2")

        super(CoFeaturesGraph, self).add_edge(node_id1, node_id2, weight=weight)
        if weight > self.max_weight:
            self.max_weight = weight
            self.max_weight_edge = temp_edge

    def get_node(self, node_id: int):
        return self._nodes.get(node_id, None)

    def get_edge(self, node_id1: int, node_id2: int):
        if node_id1 > node_id2:
            temp_node_id1 = node_id2
            temp_node_id2 = node_id1
        elif node_id1 < node_id2:
            temp_node_id1 = node_id1
            temp_node_id2 = node_id2
        else:
            raise ValueError("node_id1 should be different from node_id2")
        return self._edges.get((temp_node_id1, temp_node_id2), None)

    def get_nodes(self):
        return self._nodes

    def get_edges(self):
        return self._edges

    def get_nodes_num(self):
        return len(self._nodes)

    def get_edges_num(self):
        return len(self._edges)

    def get_max_weight_edge(self):
        return self.max_weight_edge, self.max_weight

    def all_neighbors(self, node_ids: list, sort=True) -> Tuple[List[int], List[Tuple[int, int]]]:
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


class CovisibilityNode:
    def __init__(self, node_id: int, image: np.ndarray, intrinsics: np.ndarray, all_features: FeaturesData):
        self.node_id = node_id
        self.image = image
        self.intrinsics = intrinsics
        self.all_features = all_features
        self.pose = None


class CovisibilityEdge:
    def __init__(self, node_id1: int, node_id2: int, feature1: MatchedFeatures, feature2: MatchedFeatures, cloud: np.ndarray, weight: int):
        self.node_id1 = node_id1
        self.node_id2 = node_id2
        self.feature1 = feature1
        self.feature2 = feature2
        self.cloud = cloud
        self.weight = weight


class CovisibilityGraph(nx.Graph):
    def __init__(self):
        super(CovisibilityGraph, self).__init__()
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

    def add_node(self, node_id: int, image: np.ndarray, intrinsics: np.ndarray, all_features: FeaturesData):
        self._nodes[node_id] = CovisibilityNode(node_id, image, intrinsics, all_features)
        super(CovisibilityGraph, self).add_node(node_id)

    def add_edge(self, node_id1: int, node_id2: int, feature1: MatchedFeatures, feature2: MatchedFeatures, cloud: np.ndarray, weight: int):
        temp_edge = None
        if node_id1 < node_id2:
            self._edges[(node_id1, node_id2)] = CovisibilityEdge(node_id1, node_id2, feature1, feature2, cloud, weight)
            temp_edge = self._edges[(node_id1, node_id2)]
        elif node_id1 > node_id2:
            self._edges[(node_id2, node_id1)] = CovisibilityEdge(node_id2, node_id1, feature2, feature1, cloud, weight)
            temp_edge = self._edges[(node_id2, node_id1)]
        else:
            raise ValueError("node_id1 should be different from node_id2")

        super(CovisibilityGraph, self).add_edge(node_id1, node_id2, weight=weight)
        if weight > self.max_weight:
            self.max_weight = weight
            self.max_weight_edge = temp_edge

    def get_node(self, node_id):
        return self._nodes.get(node_id, None)

    def get_edge(self, node_id1, node_id2):
        if node_id1 > node_id2:
            temp_node_id1 = node_id2
            temp_node_id2 = node_id1
        elif node_id1 < node_id2:
            temp_node_id1 = node_id1
            temp_node_id2 = node_id2
        else:
            raise ValueError("node_id1 should be different from node_id2")
        return self._edges.get((temp_node_id1, temp_node_id2), None)

    def get_nodes(self):
        return self._nodes

    def get_edges(self):
        return self._edges

    def get_nodes_num(self):
        return len(self._nodes)

    def get_edges_num(self):
        return len(self._edges)

    def get_max_weight_edge(self):
        return self.max_weight_edge, self.max_weight

    def all_neighbors(self, node_ids: list, sort=True) -> Tuple[List[int], List[Tuple[int, int]]]:
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

    def draw(self):
        nx.draw(self)
        plt.show()


_co_features_graph = CoFeaturesGraph()
_covisibility_graph = CovisibilityGraph()
