import cv2
import numpy as np
from typing import Any, Dict, Tuple, List

from .utils import *
from .feature import *
from .Initializer import *
from .CloudView import *
from .PySBA import *


def incremental_reconstruction(cloud_viewer: CloudView, cfg: Dict[str, Any] = None, fast: bool = False,
                               ba: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if cfg is None:
        cfg = load_config()

    co_features_graph = Initializer.co_features_graph
    covisibility_graph = Initializer.covisibility_graph

    node_used = set()
    total_cloud = np.empty((1, 3))
    total_colors = np.empty((1, 3))

    if fast:
        image_id1, image_id2, cloud, colors = Initializer.FastInitialize(cloud_viewer)
    else:
        image_id1, image_id2, cloud, colors = Initializer.Initialize(cloud_viewer)

    cloud_viewer = Initializer.CloudViewer

    total_cloud = np.vstack((total_cloud, cloud))
    total_colors = np.vstack((total_colors, colors))
    node_used.add(image_id1)
    node_used.add(image_id2)

    counter = 0
    while len(co_features_graph.get_edges()) > 0:
        # if counter == 2:
        #     break
        counter += 1

        neighbor_nodes, neighbor_edges = co_features_graph.all_neighbors(list(node_used))

        # 如果没有邻接边，说明没有可再用于增量重建的边了；如果所有节点都已经使用，说明所有节点都已经用于增量重建了
        if len(neighbor_edges) == 0 and len(node_used) == co_features_graph.get_nodes_num():
            break

        # ----------------- 选取已重建对应点最多的边 -----------------
        max_built_index = None
        incre_mask = None
        base_cv_edge = None
        incre_cf_edge = None
        for neighbor_edge in neighbor_edges:
            image_id1 = neighbor_edge.node_id1
            image_id2 = neighbor_edge.node_id2

            if image_id1 in node_used and image_id2 in node_used:
                pass
            else:
                temp_index = None
                temp_mask = None
                if image_id1 in node_used:
                    cv_nnodes, cv_nedges = covisibility_graph.all_neighbors([image_id1])
                    if len(cv_nedges) == 0:
                        raise ValueError("cv_nedges is empty but should not be!")
                    for cv_nedge in cv_nedges:
                        built_index = []
                        if cv_nedge.node_id1 == image_id1:
                            built_index, mask = find_built_features(cv_nedge.feature1,
                                                                    neighbor_edge.feature1.index)
                        elif cv_nedge.node_id2 == image_id1:
                            built_index, mask = find_built_features(cv_nedge.feature2,
                                                                    neighbor_edge.feature1.index)

                        if temp_index is None or len(built_index) > len(temp_index):
                            temp_index = built_index
                            temp_mask = mask
                            base_cv_edge = cv_nedge
                elif image_id2 in node_used:
                    cv_nnodes, cv_nedges = covisibility_graph.all_neighbors([image_id2])
                    if len(cv_nedges) == 0:
                        raise ValueError("cv_nedges is empty but should not be!")
                    for cv_nedge in cv_nedges:
                        built_index = []
                        if cv_nedge.node_id1 == image_id2:
                            built_index, mask = find_built_features(cv_nedge.feature1,
                                                                    neighbor_edge.feature2.index)
                        elif cv_nedge.node_id2 == image_id2:
                            built_index, mask = find_built_features(cv_nedge.feature2,
                                                                    neighbor_edge.feature2.index)

                        if temp_index is None or len(built_index) > len(temp_index):
                            temp_index = built_index
                            temp_mask = mask
                            base_cv_edge = cv_nedge
                else:
                    raise ValueError("both image_id1 and image_id2 are not in node_used!")

                if max_built_index is None or len(temp_index) > len(max_built_index):
                    max_built_index = temp_index
                    incre_mask = temp_mask
                    incre_cf_edge = neighbor_edge

        if max_built_index is None:
            break

        # ----------------- 求解PnP问题并三角化 -----------------
        image_id1 = incre_cf_edge.node_id1
        image_id2 = incre_cf_edge.node_id2

        base_id = image_id1 if image_id1 in node_used else image_id2
        incre_id = image_id2 if image_id1 in node_used else image_id1

        base_cv_node = covisibility_graph.get_node(base_id)
        incre_cv_node = covisibility_graph.get_node(incre_id)

        base_feature = None
        incre_feature = None
        if incre_id == image_id1:
            base_feature = incre_cf_edge.feature2
            incre_feature = incre_cf_edge.feature1
        elif incre_id == image_id2:
            base_feature = incre_cf_edge.feature1
            incre_feature = incre_cf_edge.feature2

        P0 = base_cv_node.pose
        base_cloud = base_cv_edge.cloud[max_built_index]
        ret, rvecs, t, inliers = cv2.solvePnPRansac(base_cloud, incre_feature.mask(incre_mask).points,
                                                    incre_cv_node.intrinsics, np.zeros((5, 1)), cv2.SOLVEPNP_EPNP)
        R, _ = cv2.Rodrigues(rvecs)
        Rt = np.hstack((R, t))
        P1 = np.matmul(incre_cv_node.intrinsics, Rt)
        incre_cv_node.pose = P1

        cloud = cv2.triangulatePoints(P0, P1, base_feature.points.T, incre_feature.points.T).T
        cloud = cv2.convertPointsFromHomogeneous(cloud)[:, 0, :]

        # 局部BA优化
        if ba:
            cameraArray = np.hstack([rvecs.reshape(1, 3), t.reshape(1, 3),
                                     np.array([incre_cv_node.intrinsics[0, 0], 0., 0.], dtype=np.float32).reshape(1,
                                                                                                                  3)])
            points3D = cloud
            cameraIndices = np.zeros(len(cloud), dtype=np.int32).ravel()
            point2DIndices = np.array([np.arange(len(cloud))], dtype=np.int32).ravel()
            points2D = incre_feature.points
            print("local BA start!")
            local_ba = PySBA(cameraArray=cameraArray, points3D=points3D, points2D=points2D, cameraIndices=cameraIndices,
                             point2DIndices=point2DIndices)
            camera_params, cloud = local_ba.bundleAdjust()
            print("local BA finished!")

            incre_cv_node.pose = np.hstack((cv2.Rodrigues(camera_params[0][0:3].reshape(3, 1))[0],
                                            camera_params[0][3:6].reshape(3, 1)))
            total_cloud = np.vstack((total_cloud, cloud))
            total_colors = np.vstack((total_colors, incre_feature.colors))

        else:
            total_cloud = np.vstack((total_cloud, cloud))
            total_colors = np.vstack((total_colors, incre_feature.colors))

        covisibility_graph.add_edge(base_id, incre_id, base_feature, incre_feature, cloud, len(cloud))

        node_used.add(image_id1)
        node_used.add(image_id2)
        co_features_graph.del_edge(image_id1, image_id2)

        cloud_viewer.update(incre_cv_node.image, Rt, incre_cv_node.intrinsics, total_cloud, total_colors)
        print("Built cloud based on image {} and image {}.".format(base_id, incre_id))

    return total_cloud, total_colors


def find_built_features(built_features: MatchedFeatures, queried_index: List) -> Tuple[List, np.ndarray]:
    built_set = set(built_features.index)
    mask_list = []
    built_index = []
    for item in queried_index:
        if item in built_set:
            temp_index = built_features.index.index(item)
            built_index.append(temp_index)
            mask_list.append(True)
        else:
            mask_list.append(False)
    mask = np.array(mask_list)

    return built_index, mask
