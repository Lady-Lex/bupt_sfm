import os
import yaml
from dataclasses import dataclass, asdict
from typing import Any, Dict, IO, Union

from .api import *

@dataclass
class SfMConfig:
    ##################################
    # Params for running
    ##################################
    running_image_dir: str = ""
    running_calib: str = ""
    running_stride: int = 1
    running_skip: int = 0
    running_viz: bool = True
    running_ros: bool = False
    running_robust: bool = True
    running_save_reconstruction: bool = True

    ##################################
    # Params for image
    ##################################
    image_height: int = 2048
    image_width: int = 3072
    image_downscale: int = 2

    ##################################
    # Params for features matching
    ##################################
    feature_matched_min: int = 100
    feature_distance_ratio: float = 0.7

    ##################################
    # Params for features detection
    ##################################
    # If true, apply square root mapping to features
    feature_root: bool = True
    # If fewer frames are detected, sift_peak_threshold/surf_hessian_threshold is reduced.
    feature_min_frames: int = 4000
    # Same as above but for panorama images
    feature_min_frames_panorama: int = 16000
    # Resize the image if its size is larger than specified. Set to -1 for original size
    feature_process_size: int = 2048
    # Same as above but for panorama images
    feature_process_size_panorama: int = 4096
    feature_use_adaptive_suppression: bool = False

    ##################################
    # Params for SIFT
    ##################################
    # Smaller value -> more features
    sift_peak_threshold: float = 0.1
    sift_edge_threshold: int = 10

    ##################################
    # Params for BA
    ##################################
    BA_activated: bool = False

    ##################################
    # Params for dpviewer
    ##################################
    dpvierwer_activated: bool = True

    ##################################
    # Params for save
    ##################################
    save_directory: str = ""
    save_distance_thresh: float = 300.0


def default_config() -> Dict[str, Any]:
    """Return default configuration"""
    return asdict(SfMConfig())


def load_config(filepath="") -> Dict[str, Any]:
    """DEPRECATED: = Load config from a config.yaml filepath"""
    if not os.path.isfile(filepath):
        return default_config()

    with open(filepath) as fin:
        return load_config_from_fileobject(fin)


def load_config_from_fileobject(
    f: Union[IO[bytes], IO[str], bytes, str]
) -> Dict[str, Any]:
    """Load config from a config.yaml fileobject"""
    config = default_config()

    new_config = yaml.safe_load(f)
    if new_config:
        for k, v in new_config.items():
            config[k] = v

    return config

def load_config_from_api() -> Dict[str, Any]:
    if api.config_dict is None:
        config = default_config()
    else:
        config = default_config()
        for k, v in api.config_dict.items():
            config[k] = v

    return config