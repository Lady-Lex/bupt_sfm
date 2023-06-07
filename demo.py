import os
from multiprocessing import Process, Queue

from bupt_sfm.sfm_main import *
from bupt_sfm.stream import *
from bupt_sfm.config import *


def run(config: str) -> None:
    cfg = load_config(config)

    queue = Queue(maxsize=10)

    if cfg["running_ros"]:
        reader = Process(target=run_ros_topic_stream, args=(cfg, queue))
    else:
        if os.path.isdir(cfg["running_image_dir"]):
            reader = Process(target=image_stream, args=(cfg, queue))
        else:
            raise Exception("Wrong image directory path!")

    print("SFM Start!")
    reader.start()

    sfm = sfm_runner(queue, cfg)
    _, _ = sfm()

    reader.join()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./config/default.yaml", help="Path to config file")

    args = parser.parse_args()

    run(args.config)
