import sys
import copy
from tqdm import tqdm

from .utils import *
from .feature import *
from .reconstruction import incremental_reconstruction
from .graph import _co_features_graph, _covisibility_graph


class sfm_runner(object):
    def __init__(self, queue):
        self.queue = queue

    def __call__(self, fast=False, viz=False):
        while True:
            (t, image, intrinsics) = self.queue.get()
            if t < 0:
                break

            # 向共特征点图和共视关系图中添加节点
            _co_features_graph.add_node(t, image, intrinsics)
            _covisibility_graph.add_node(t, image, intrinsics)

        if _co_features_graph.get_nodes_num() < 2:
            raise Exception("Less than 2 images to reconstruct!")

        image_nodes = _co_features_graph.get_nodes()
        if fast:
            for node_id1, image1 in tqdm(image_nodes.items()):
                for node_id2, image2 in image_nodes.items():
                    if node_id2 == node_id1 + 1:
                        # 计算两个节点的共视特征点
                        feature1, feature2, E = SIFTmatcher.search_by_Initialization(image1.image, image2.image, intrinsics)
                        weight = len(feature1.points)
                        # 参数：如果共视特征点数量小于100，则跳过
                        if weight < 100:
                            continue
                        # 向共特征点图中添加边
                        _co_features_graph.add_edge(node_id1, node_id2, feature1, feature2, weight, E)
        else:
            # 从共特征点图中获取节点
            for node_id1, image1 in tqdm(image_nodes.items()):
                for node_id2, image2 in image_nodes.items():
                    if node_id1 >= node_id2:
                        continue
                    # 计算两个节点的共视特征点
                    feature1, feature2, E = SIFTmatcher.search_by_Initialization(image1.image, image2.image, intrinsics)
                    weight = len(feature1.points)
                    # 参数：如果共视特征点数量小于100，则跳过
                    if weight < 100:
                        continue
                    # 向共特征点图中添加边
                    _co_features_graph.add_edge(node_id1, node_id2, feature1, feature2, weight, E)

        # 可视化共特征点图
        _co_features_graph.draw()
        
        # 增量式重建
        total_cloud, total_color = incremental_reconstruction(fast=False, viz=viz)

        # 可视化共视关系图
        _covisibility_graph.draw()

        return total_cloud, total_color

    def reset(self):
        pass
