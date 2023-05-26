import cv2
import numpy as np

from .feature import *
from .graph import co_features_graph, CoFeaturesGraph


class Initializer:
    feature_graph: CoFeaturesGraph = co_features_graph

    @classmethod
    def Initialize(cls, K):
        feature_graph = cls.feature_graph
        images_num = len(feature_graph.get_nodes())
        # 1. 选取最大权重的边
        max_weight_edge, max_weight = feature_graph.get_max_weight_edge()

        # 2. 初始三角化
        I0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
        camera_poses = np.array(I0.ravel())
        P0 = np.matmul(K, I0)

        E = feature_graph.get_edge(max_weight_edge.node_id1, max_weight_edge.node_id2).EssentialMatrix
        _, R, t, mask = cv2.recoverPose(E, max_weight_edge.feature1.points, max_weight_edge.feature2.points, K)
        t = t.reshape(3, 1)
        Rt = np.hstack((R, t))
        P1 = np.matmul(K, Rt)
        cloud = cv2.triangulatePoints(P0, P1,  max_weight_edge.feature1.points.T,  max_weight_edge.feature2.points.T).T
        cloud = cv2.convertPointsFromHomogeneous(cloud)[:, 0, :]

        return cloud
