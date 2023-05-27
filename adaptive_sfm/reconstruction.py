import cv2
import numpy as np

from .feature import *
from .Initializer import *


def incremental_reconstruction():
    co_features_graph = Initializer.feature_graph
    node_used = set()
    total_cloud = np.zeros((1, 3))
    # total_color: 3D点云颜色
    total_color = np.zeros((1, 3))

    image_id0, image_id1, cloud = Initialize.Initialize()
    total_cloud = np.vstack((total_cloud, cloud))

    node_used.add(image_id0)
    node_used.add(image_id1)
    # 选取特征点与已重建点的交集最多的边
    while len(co_features_graph.get_edges()) > 0:
        neighbor_nodes, neighbor_edges = co_features_graph.all_neighbors()
        for image_id0, image_id1 in neighbor_edges:
            # if image_id0 in node_used and image_id1 in node_used:
            #     co_features_graph.del_edge(image_id0, image_id1)
            # else:
            edge = co_features_graph.get_edge(image_id0, image_id1)
            feature1 = edge.feature1
            feature2 = edge.feature2
            base_points = feature1.points if image_id0 in node_used else feature2.points
            incre_points = feature2.points if image_id0 in node_used else feature1.points
            P0 = edge.node_id1.pose if image_id0 in node_used else edge.node_id2.pose
            ret, rvecs, t, inliers = cv2.solvePnPRansac(cloud, incre_points, edge.node_id1.intrinsics, cv2.SOLVEPNP_EPNP)
            R, _ = cv2.Rodrigues(rvecs)
            Rt = np.hstack((R, t))
            P1 = np.matmul(K, Rt)
            cloud = cv2.triangulatePoints(P0, P1, base_points.T, incre_points.T).T
            cloud = cv2.convertPointsFromHomogeneous(cloud)[:, 0, :]

            total_cloud = np.vstack((total_cloud, cloud))
            # 局部BA优化

            node_used.add(image_id0)
            node_used.add(image_id1)
            co_features_graph.del_edge(image_id0, image_id1)
