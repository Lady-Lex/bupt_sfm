import cv2
import numpy as np
from typing import Any, Dict

from .feature import *
from .graph import _co_features_graph, _covisibility_graph, CoFeaturesGraph, CovisibilityGraph
from .CloudView import *


class Initializer:
    cfg: Dict[str, Any] = load_config()
    co_features_graph: CoFeaturesGraph = _co_features_graph
    covisibility_graph: CovisibilityGraph = _covisibility_graph
    CloudViewer: CloudView

    @classmethod
    def Initialize(cls, cloud_viewer: CloudView):
        co_features_graph = cls.co_features_graph
        covisibility_graph = cls.covisibility_graph
        cls.CloudViewer = cloud_viewer

        # 1. 选取最大权重的边
        max_weight_edge, max_weight = co_features_graph.get_max_weight_edge()

        if max_weight_edge is None:
            raise ValueError(
                "max_weight_edge is None, meaning that there isn't enough matched features for images each other to "
                "initialize!")

        node_1 = co_features_graph.get_node(max_weight_edge.node_id1)
        node_2 = co_features_graph.get_node(max_weight_edge.node_id2)

        assert (node_1.intrinsics == node_2.intrinsics).all()

        K = node_1.intrinsics
        # 2. 初始三角化
        I0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
        P0 = np.matmul(K, I0)

        E = max_weight_edge.EssentialMatrix
        _, R, t, mask = cv2.recoverPose(E, max_weight_edge.feature1.points, max_weight_edge.feature2.points, K)
        t = t.reshape(3, 1)
        Rt = np.hstack((R, t))
        P1 = np.matmul(K, Rt)

        cloud = cv2.triangulatePoints(P0, P1, max_weight_edge.feature1.points.T, max_weight_edge.feature2.points.T).T
        cloud = cv2.convertPointsFromHomogeneous(cloud)[:, 0, :]
        colors = max_weight_edge.feature2.colors
        co_features_graph.del_edge(max_weight_edge.node_id1, max_weight_edge.node_id2)

        # 维护共视关系图
        covisibility_graph.get_node(max_weight_edge.node_id1).pose = P0
        covisibility_graph.get_node(max_weight_edge.node_id2).pose = P1
        covisibility_graph.add_edge(max_weight_edge.node_id1, max_weight_edge.node_id2, max_weight_edge.feature1,
                                    max_weight_edge.feature2, cloud, max_weight)

        cls.CloudViewer.update(node_1.image, I0, node_1.intrinsics, cloud, colors)
        cls.CloudViewer.update(node_2.image, Rt, node_2.intrinsics, cloud, colors)

        print("Initialization finished! Built cloud based on image {} and image {}.".format(max_weight_edge.node_id1,
                                                                                            max_weight_edge.node_id2))
        return max_weight_edge.node_id1, max_weight_edge.node_id2, cloud, colors

    @classmethod
    def FastInitialize(cls, cloud_viewer: CloudView):
        co_features_graph = cls.co_features_graph
        covisibility_graph = cls.covisibility_graph
        cls.CloudViewer = cloud_viewer

        node_id1 = 0
        node_id2 = 1
        first_edge = co_features_graph.get_edge(node_id1, node_id2)
        while first_edge is None:
            node_id1 += 1
            node_id2 += 1
            first_edge = co_features_graph.get_edge(node_id1, node_id2)
            if node_id2 >= len(co_features_graph.get_nodes()):
                raise Exception("Not enough matched features for initialization!")
        first_weight = first_edge.weight

        node_1 = co_features_graph.get_node(0)
        node_2 = co_features_graph.get_node(1)

        assert (node_1.intrinsics == node_2.intrinsics).all()

        K = node_1.intrinsics
        I0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
        P0 = np.matmul(K, I0)

        E = first_edge.EssentialMatrix
        _, R, t, mask = cv2.recoverPose(E, first_edge.feature1.points, first_edge.feature2.points, K)
        t = t.reshape(3, 1)
        Rt = np.hstack((R, t))
        P1 = np.matmul(K, Rt)

        cloud = cv2.triangulatePoints(P0, P1, first_edge.feature1.points.T, first_edge.feature2.points.T).T
        cloud = cv2.convertPointsFromHomogeneous(cloud)[:, 0, :]
        colors = first_edge.feature2.colors
        co_features_graph.del_edge(first_edge.node_id1, first_edge.node_id2)

        # 维护共视关系图
        covisibility_graph.get_node(first_edge.node_id1).pose = P0
        covisibility_graph.get_node(first_edge.node_id2).pose = P1
        covisibility_graph.add_edge(first_edge.node_id1, first_edge.node_id2, first_edge.feature1,
                                    first_edge.feature2, cloud, first_weight)

        cls.CloudViewer.update(node_1.image, I0, node_1.intrinsics, cloud, colors)
        cls.CloudViewer.update(node_2.image, Rt, node_2.intrinsics, cloud, colors)

        print("Initialization finished! Built cloud based on image {} and image {}.".format(first_edge.node_id1,
                                                                                            first_edge.node_id2))
        return first_edge.node_id1, first_edge.node_id2, cloud, colors
