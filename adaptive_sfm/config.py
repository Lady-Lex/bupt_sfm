import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, IO, Union


@dataclass
class SfMConfig:
    ##################################
    # Params for image
    ##################################
    image_height: int = 1024
    image_width: int = 1536

    ##################################
    # Params for features matching
    ##################################
    feature_matched_min: int = 100

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
    # See OpenCV doc
    sift_edge_threshold: int = 10


def load_config() -> Dict[str, Any]:
    """Return default configuration"""
    return asdict(SfMConfig())
