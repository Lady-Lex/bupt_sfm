import numpy as np
from typing import Any, Dict

try:
    import torch
    from dpviewer import Viewer
    use_dpviewer = True
except ModuleNotFoundError as e:
    use_dpviewer = False

from .utils import T_to_seven_numbers
from .config import load_config


class CloudView:
    def __init__(self, cfg: Dict[str, Any] = None, ht: int = 480, wd: int = 640, viz: bool = True):
        if not use_dpviewer:
            self.viewer = None
            return

        if cfg is None:
            cfg = load_config()

        self.cfg = cfg

        self.n = 0  # number of frames
        self.m = 0  # number of patches
        # self.M = self.cfg.PATCHES_PER_FRAME
        # self.N = self.cfg.BUFFER_SIZE
        self.N = 50
        self.M = 1000

        self.ht = ht  # image height
        self.wd = wd  # image width

        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")
        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.poses_[:, 6] = 1.0

        self.viewer = None
        if viz:
            self.start_viewer()

    def start_viewer(self):
        # intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.poses_,
            self.points_,
            self.colors_,
            self.intrinsics_)

    def update(self, image: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray, points: np.ndarray, colors: np.ndarray):
        if not use_dpviewer:
            return

        self.image_ = torch.from_numpy(image).to(torch.uint8).to("cpu")
        if self.viewer is not None:
            self.viewer.update_image(self.image_)

        self.poses_[self.n] = torch.from_numpy(T_to_seven_numbers(pose)).to(torch.float32).to("cuda")
        K = np.array([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]])

        self.intrinsics_[self.n] = torch.from_numpy(K).to(torch.float32).to("cuda")
        self.points_[:len(points)] = torch.from_numpy(points).to(torch.float).to("cuda")
        self.colors_[self.n] = torch.from_numpy(colors[0:self.M]).to(torch.uint8).to("cuda")
        self.n += 1
        self.m += self.M



