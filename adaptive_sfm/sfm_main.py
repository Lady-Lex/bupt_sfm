import sys
import copy

from .feature import *
from .reconstruction import incremental_reconstruction
from .graph import _co_features_graph


class sfm_runner(object):
    def __init__(self, queue):
        self.queue = copy.deepcopy(queue)
        self.run()

    def run(self):
        while True:
            (t, image, intrinsics) = self.queue.get()
            if t < 0:
                break

            # 向共特征点图中添加节点
            _co_features_graph.add_node(t, image, intrinsics)

        # 从共特征点图中获取节点
        image_nodes = _co_features_graph.get_nodes()
        for node_id1, image1 in image_nodes.items():
            for node_id2, image2 in image_nodes.items():
                if node_id1 >= node_id2:
                    continue
                # 计算两个节点的共视特征点
                feature1, feature2, E = SIFTmatcher.search_by_Initialization(image1, image2, intrinsics)
                weight = len(feature1.points)
                # 参数：如果共视特征点数量小于100，则跳过
                if weight < 100:
                    continue
                # 向共特征点图中添加边
                _co_features_graph.add_edge(node_id1, node_id2, feature1, feature2, weight, E)

        # 增量式重建
        # incremental_reconstruction()

    def reset(self):
        pass

    # @staticmethod
    # def start_viewer():
        # from dpviewer import Viewer
        # pass
