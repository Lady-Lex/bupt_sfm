import networkx as nx
import matplotlib.pyplot as plt  # 导入模块函数

from .feature import *


class CoFeaturesNode:
    def __init__(self, node_id, image):
        self.node_id = node_id
        self.image = image
        self.pose = None


class CoFeaturesEdge:
    def __init__(self, node_id1, node_id2, weight, feature1, feature2, E):
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

    def add_node(self, node_id, image):
        self._nodes[node_id] = CoFeaturesNode(node_id, image)
        super(CoFeaturesGraph, self).add_node(node_id)

    def add_edge(self, node_id1, node_id2, weight, feature1: object, feature2, E):
        self._edges[(node_id1, node_id2)] = CoFeaturesEdge(node_id1, node_id2, weight, feature1, feature2, E)
        super(CoFeaturesGraph, self).add_edge(node_id1, node_id2, weight=weight)
        if weight > self.max_weight:
            self.max_weight = weight
            self.max_weight_edge = self._edges[(node_id1, node_id2)]

    def get_node(self, node_id):
        return self._nodes[node_id]

    def get_edge(self, node_id1, node_id2):
        return self._edges[(node_id1, node_id2)]

    def get_nodes(self):
        return self._nodes

    def get_edges(self):
        return self._edges

    def get_max_weight_edge(self):
        return self.max_weight_edge, self.max_weight

    def draw(self):
        nx.draw(self)
        plt.show()


class CovisibilityGraph(nx.Graph):
    def __init__(self):
        super(CovisibilityGraph, self).__init__()
        self._nodes = {}
        self._edges = {}

    def reset(self):
        self._nodes = {}
        self._edges = {}
        self.clear()

    def add_node(self, node_id, node):
        self._nodes[node_id] = node
        super(CovisibilityGraph, self).add_node(node_id)

    def add_edge(self, node_id1, node_id2, weight):
        self._edges[(node_id1, node_id2)] = weight
        super(CovisibilityGraph, self).add_edge(node_id1, node_id2, weight=weight)

    def get_node(self, node_id):
        return self._nodes[node_id]

    def get_edge(self, node_id1, node_id2):
        return self._edges[(node_id1, node_id2)]

    def get_nodes(self):
        return self._nodes

    def get_edges(self):
        return self._edges

    def draw(self):
        nx.draw(self)
        plt.show()


co_features_graph = CoFeaturesGraph()
covisibility_graph = CovisibilityGraph()
