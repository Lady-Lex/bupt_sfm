import cv2
import numpy as np
import os
from pathlib import Path
from multiprocessing import Process, Queue
from plyfile import PlyElement, PlyData

from adaptive_sfm.sfm_main import *
from adaptive_sfm.stream import image_stream, video_stream
from adaptive_sfm.utils import *

debug = True


def run(image_dir, calib, stride=1, skip=0, viz=False, ros=False, save_reconstruction=False):
    queue = Queue(maxsize=8)

    if os.path.isdir(image_dir):
        reader = Process(target=image_stream, args=(queue, image_dir, calib, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, image_dir, calib, stride, skip))

    reader.start()

    sfm = sfm_runner(queue)
    total_cloud, total_color = sfm(fast=True, viz=viz)

    reader.join()
    if save_reconstruction:
        to_ply(total_cloud, total_color, "pointcloud/sfm_output.ply")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default="../data/fountain-P11/images")
    parser.add_argument('--calib', type=str, default="../data/fountain-P11/images/intrinsics.txt")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--skip', type=int, default=0)
    # parser.add_argument('--plot', action="store_true")
    # parser.add_argument('--buffer', type=int, default=2048)
    parser.add_argument('--config', default="config/default.yaml")
    # parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--ros', action="store_true")
    # parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_reconstruction', action="store_true")
    # parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()

    # cfg.merge_from_file(args.config)
    # cfg.BUFFER_SIZE = args.buffer

    # print("Running with config...")
    # print(cfg)

    run(args.image_dir, args.calib, args.stride, args.skip, args.viz, args.ros, args.save_reconstruction)

    # name = Path(args.image_dir).stem
    #
    # if args.save_reconstruction:
    #     pred_traj, ply_data = pred_traj
    #     ply_data.write(f"{name}.ply")
    #     print(f"Saved {name}.ply")
    #
    # if args.save_trajectory:
    #     Path("saved_trajectories").mkdir(exist_ok=True)
    #     save_trajectory_tum_format(pred_traj, f"saved_trajectories/{name}.txt")
    #
    # if args.plot:
    #     Path("trajectory_plots").mkdir(exist_ok=True)
    #     plot_trajectory(pred_traj, title=f"DPVO Trajectory Prediction for {name}", filename=f"trajectory_plots/{name}.pdf")
