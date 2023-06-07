import time
from tqdm import tqdm
from typing import Any, Dict, Tuple
from multiprocessing import Queue

from .utils import *
from .feature import *
from .config import *
from .reconstruction import incremental_reconstruction
from .graph import _co_features_graph, _covisibility_graph
from .CloudView import CloudView


class sfm_runner(object):
    def __init__(self, queue: Queue, cfg: Dict[str, Any]) -> None:
        self.queue = queue
        self.cfg = cfg
        self.cloud_viewer = CloudView(cfg=self.cfg)

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        print("Receiving images and extracting features...")
        while True:
            (t, image, intrinsics) = self.queue.get()
            if t < 0:
                print("All images processed and received!")
                break

            all_features = extract_features(image, self.cfg, False)
            # 向共特征点图和共视关系图中添加节点
            _co_features_graph.add_node(t, image, intrinsics, all_features)
            _covisibility_graph.add_node(t, image, intrinsics, all_features)

        if _co_features_graph.get_nodes_num() < 2:
            raise Exception("Less than 2 images to reconstruct!")

        print("Building co-features graph...")
        image_nodes = _co_features_graph.get_nodes()

        # 从共特征点图中获取节点
        for node_id1, image1 in image_nodes.items():
            for node_id2, image2 in image_nodes.items():
                if self.cfg["running_robust"]:
                    if node_id1 >= node_id2:
                        continue
                else:
                    if node_id2 != node_id1 + 1:
                        continue
                # 计算两个节点的共视特征点
                feature1, feature2, E = SIFTmatcher.search_by_Initialization(self.cfg, image1.all_features,
                                                                             image2.all_features, intrinsics)
                weight = len(feature1.points)
                # 参数：如果共视特征点数量小于100，则跳过
                if weight < self.cfg["feature_matched_min"]:
                    continue
                # 向共特征点图中添加边
                _co_features_graph.add_edge(node_id1, node_id2, feature1, feature2, weight, E)

        # 可视化共特征点图
        # _co_features_graph.draw()

        # 增量式重建
        start_time = time.time()
        total_cloud, total_color = incremental_reconstruction(cloud_viewer=self.cloud_viewer, cfg=self.cfg)
        print("Incremental reconstruction time: ", time.time() - start_time)

        # 可视化共视关系图
        # _covisibility_graph.draw()

        if self.cfg["running_save_reconstruction"]:
            save_ply(self.cfg, total_cloud, total_color)

        print("SFM Done!")

        # if self.cloud_viewer.viewer is not None:
        #     self.cloud_viewer.viewer.join()

        return total_cloud, total_color
