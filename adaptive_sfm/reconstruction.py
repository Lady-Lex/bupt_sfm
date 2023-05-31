import cv2
import numpy as np

from .utils import *
from .feature import *
from .Initializer import *
from .CloudView import *


def incremental_reconstruction(config=None, fast: bool = False, viz: bool = False):
    if config is None:
        config = load_config()
    co_features_graph = Initializer.co_features_graph
    covisibility_graph = Initializer.covisibility_graph

    node_used = set()
    total_cloud = np.empty((1, 3))
    total_colors = np.empty((1, 3))

    if fast:
        image_id1, image_id2, cloud, colors = Initializer.FastInitialize(viz)
    else:
        image_id1, image_id2, cloud, colors = Initializer.Initialize(viz)

    CloudViewer = Initializer.CloudViewer

    total_cloud = np.vstack((total_cloud, cloud))
    total_colors = np.vstack((total_colors, colors))
    node_used.add(image_id1)
    node_used.add(image_id2)

    while len(co_features_graph.get_edges()) > 0:
        neighbor_nodes, neighbor_edges = co_features_graph.all_neighbors(list(node_used))

        # 如果没有邻接边，说明没有可再用于增量重建的边了；如果所有节点都已经使用，说明所有节点都已经用于增量重建了
        if len(neighbor_edges) == 0 and len(node_used) == co_features_graph.get_nodes_num():
            break

        # ----------------- 选取已重建对应点最多的边 -----------------
        max_built_index = None
        base_cv_edge = None
        incre_cf_edge = None
        for neighbor_edge in neighbor_edges:
            image_id1 = neighbor_edge.node_id1
            image_id2 = neighbor_edge.node_id2

            if image_id1 in node_used and image_id2 in node_used:
                pass
            else:
                if image_id1 in node_used:
                    cv_nnodes, cv_nedges = covisibility_graph.all_neighbors([image_id1])
                    if len(cv_nedges) == 0:
                        raise ValueError("cv_nedges is empty but should not be")
                    temp_index = None
                    for cv_nedge in cv_nedges:
                        if cv_nedge.node_id1 == image_id1:
                            built_index = get_rows_index(neighbor_edge.feature1.points, cv_nedge.feature1.points)
                            if temp_index is None or len(built_index) > len(temp_index):
                                temp_index = built_index
                                base_cv_edge = cv_nedge
                        elif cv_nedge.node_id2 == image_id1:
                            built_index = get_rows_index(neighbor_edge.feature1.points, cv_nedge.feature2.points)
                            if temp_index is None or len(built_index) > len(temp_index):
                                temp_index = built_index
                                base_cv_edge = cv_nedge
                elif image_id2 in node_used:
                    cv_nnodes, cv_nedges = covisibility_graph.all_neighbors([image_id2])
                    if len(cv_nedges) == 0:
                        raise ValueError("cv_nedges is empty but should not be")
                    temp_index = None
                    for cv_nedge in cv_nedges:
                        if cv_nedge.node_id1 == image_id2:
                            built_index = get_rows_index(neighbor_edge.feature2.points, cv_nedge.feature1.points)
                            if temp_index is None or len(built_index) > len(temp_index):
                                temp_index = built_index
                                base_cv_edge = cv_nedge
                        elif cv_nedge.node_id2 == image_id2:
                            built_index = get_rows_index(neighbor_edge.feature2.points, cv_nedge.feature2.points)
                            if temp_index is None or len(built_index) > len(temp_index):
                                temp_index = built_index
                                base_cv_edge = cv_nedge
                else:
                    raise ValueError("both image_id1 and image_id2 are not in node_used")

                if max_built_index is None or len(temp_index) > len(max_built_index):
                    max_built_index = temp_index
                    incre_cf_edge = neighbor_edge

        if max_built_index is None:
            break
        cloud_index = list(filter(lambda x: x is not None, max_built_index))
        point_index = np.array([True if x is not None else False for x in max_built_index])

        # ----------------- 求解PnP问题并三角化 -----------------
        image_id1 = incre_cf_edge.node_id1
        image_id2 = incre_cf_edge.node_id2

        base_id = image_id1 if image_id1 in node_used else image_id2
        incre_id = image_id2 if image_id1 in node_used else image_id1

        base_cv_node = covisibility_graph.get_node(base_id)
        incre_cv_node = covisibility_graph.get_node(incre_id)
        if incre_id == image_id1:
            base_feature = incre_cf_edge.feature2
            incre_feature = incre_cf_edge.feature1
        elif incre_id == image_id2:
            base_feature = incre_cf_edge.feature1
            incre_feature = incre_cf_edge.feature2

        P0 = base_cv_node.pose
        base_cloud = base_cv_edge.cloud[cloud_index]

        ret, rvecs, t, inliers = cv2.solvePnPRansac(base_cloud, incre_feature.points[point_index],
                                                    incre_cv_node.intrinsics, np.zeros((5, 1)), cv2.SOLVEPNP_EPNP)
        R, _ = cv2.Rodrigues(rvecs)
        Rt = np.hstack((R, t))
        P1 = np.matmul(incre_cv_node.intrinsics, Rt)
        if image_id1 in node_used:
            incre_cv_node.pose = P1
        elif image_id2 in node_used:
            incre_cv_node.pose = P1

        cloud = cv2.triangulatePoints(P0, P1, base_feature.points.T, incre_feature.points.T).T
        cloud = cv2.convertPointsFromHomogeneous(cloud)[:, 0, :]
        total_cloud = np.vstack((total_cloud, cloud))
        total_colors = np.vstack((total_colors, incre_feature.colors))
        covisibility_graph.add_edge(base_id, incre_id, base_feature, incre_feature, cloud, len(cloud))

        # 局部BA优化

        node_used.add(image_id1)
        node_used.add(image_id2)
        co_features_graph.del_edge(image_id1, image_id2)

        CloudViewer.update(incre_cv_node.image, Rt, incre_cv_node.intrinsics, total_cloud, total_colors)
        print("Built cloud based on image {} and image {}".format(base_id, incre_id))

    if CloudViewer.viewer is not None:
        CloudViewer.viewer.join()

    return total_cloud, total_colors
