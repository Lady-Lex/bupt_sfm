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

    image_id0, image_id1, cloud = Initializer.Initialize()
    total_cloud = np.vstack((total_cloud, cloud))

    node_used.add(image_id0)
    node_used.add(image_id1)
    # 选取特征点与已重建点的交集最多的边
    while len(co_features_graph.get_edges()) > 0:
        neighbor_nodes, neighbor_edges = co_features_graph.all_neighbors(list(node_used))
        if len(neighbor_edges) == 0:
            break
        for neighbor_edge in neighbor_edges:
            image_id0 = neighbor_edge.node_id1
            image_id1 = neighbor_edge.node_id2
            
            # if image_id0 in node_used and image_id1 in node_used:
            #     co_features_graph.del_edge(image_id0, image_id1)
            # else:
            base_points = neighbor_edge.feature1.points if image_id0 in node_used else neighbor_edge.feature2.points
            incre_points = neighbor_edge.feature2.points if image_id0 in node_used else neighbor_edge.feature1.points

            node_1 = co_features_graph.get_node(neighbor_edge.node_id1)
            node_2 = co_features_graph.get_node(neighbor_edge.node_id2)

            P0 = node_1.pose if image_id0 in node_used else node_2.pose
            print(cloud.shape)
            print("incre_points.shape:", incre_points.shape)
            ret, rvecs, t, inliers = cv2.solvePnPRansac(cloud, incre_points, node_1.intrinsics, np.zeros((5,1)), cv2.SOLVEPNP_EPNP)
            R, _ = cv2.Rodrigues(rvecs)
            Rt = np.hstack((R, t))
            P1 = np.matmul(K, Rt)
            if image_id0 in node_used:
                node_2.pose = P1
            elif image_id1 in node_used:
                node_1.pose = P1

            cloud = cv2.triangulatePoints(P0, P1, base_points.T, incre_points.T).T
            cloud = cv2.convertPointsFromHomogeneous(cloud)[:, 0, :]

            total_cloud = np.vstack((total_cloud, cloud))
            # 局部BA优化

            node_used.add(image_id0)
            node_used.add(image_id1)
            co_features_graph.del_edge(image_id0, image_id1)
