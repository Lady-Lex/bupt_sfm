import os
import cv2
import numpy as np
from math import sqrt
from typing import Any, Dict, Tuple
from plyfile import PlyElement, PlyData


def img_downscale(img: np.ndarray, downscale: int):
    downscale = int(downscale / 2)
    i = 1
    while i <= downscale:
        img = cv2.pyrDown(img)
        i = i + 1
    return img


def T_to_seven_numbers(T: np.ndarray):
    t = T[:3, 3]

    # 计算四元数
    R = T[:3, :3]
    w = sqrt(1 + R[0][0] + R[1][1] + R[2][2]) / 2
    x = (R[2][1] - R[1][2]) / (4 * w)
    y = (R[0][2] - R[2][0]) / (4 * w)
    z = (R[1][0] - R[0][1]) / (4 * w)
    q = np.array([x, y, z, w])
    pose = np.hstack([t, q])
    return pose


def save_ply(cfg: Dict[str, Any], point_cloud: np.ndarray, colors: np.ndarray):
    if cfg["save_directory"] == "":
        save_dir = os.path.join(os.getcwd(), "pointcloud")
    else:
        save_dir = cfg["save_directory"]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    path = os.path.join(save_dir, "sfm_output.ply")
    print(f'Saving point cloud to {path}')

    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    vertexs = np.hstack([out_points, out_colors])

    # cleaning point cloud
    mean = np.mean(vertexs[:, :3], axis=0)
    temp = vertexs[:, :3] - mean
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    v_index = np.where(dist < np.mean(dist) + cfg["save_distance_thresh"])
    vertexs = vertexs[v_index]

    cloud = vertexs[:, :3]
    colors = vertexs[:, 3:]
    points = np.array([(x, y, z, b, g, r) for (x, y, z), (b, g, r) in zip(cloud, colors)],
                      dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('blue', 'u1'), ('green', 'u1'),
                             ('red', 'u1')])
    el = PlyElement.describe(points, 'vertex', {'some_property': 'f8'}, {'some_property': 'u4'})
    ply_data = PlyData([el], text=True)

    ply_data.write(path)
    print(f"Ply file Saved, at {path}")
