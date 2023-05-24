import cv2
import time
import logging
import numpy as np
from typing import Any, Tuple, Dict, Optional

logger: logging.Logger = logging.getLogger(__name__)


class FeaturesData:
    points: np.ndarray
    descriptors: Optional[np.ndarray]
    colors: np.ndarray

    def __init__(self, points: np.ndarray, descriptors: Optional[np.ndarray], colors: np.ndarray):
        self.points = points
        self.descriptors = descriptors
        self.colors = colors

    def mask(self, mask: np.ndarray) -> "FeaturesData":
        return FeaturesData(
            self.points[mask],
            self.descriptors[mask] if self.descriptors is not None else None,
            self.colors[mask]
        )


def resized_image(image: np.ndarray, max_size: int) -> np.ndarray:
    """Resize image to feature_process_size."""
    h, w = image.shape[:2]
    size = max(w, h)
    if 0 < max_size < size:
        dsize = w * max_size // size, h * max_size // size
        return cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_AREA)
    else:
        return image


def root_feature(desc: np.ndarray, l2_normalization: bool = False) -> np.ndarray:
    if l2_normalization:
        s2 = np.linalg.norm(desc, axis=1)
        desc = (desc.T / s2).T
    s = np.sum(desc, 1)
    desc = np.sqrt(desc.T / s).T
    return desc


def extract_features_sift(
        image: np.ndarray, config: Dict[str, Any], features_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    sift_edge_threshold = config["sift_edge_threshold"]
    sift_peak_threshold = float(config["sift_peak_threshold"])
    try:
        detector = cv2.xfeatures2d.SIFT_create(
            edgeThreshold=sift_edge_threshold, contrastThreshold=sift_peak_threshold
        )
    except AttributeError as ae:
        if "no attribute 'xfeatures2d'" in str(ae):
            logger.error(
                "OpenCV Contrib modules are required to extract SIFT features"
            )
        raise
    descriptor = detector

    while True:
        logger.debug("Computing sift with threshold {0}".format(sift_peak_threshold))
        t = time.time()
        detector = cv2.xfeatures2d.SIFT_create(
            edgeThreshold=sift_edge_threshold, contrastThreshold=sift_peak_threshold
        )

        points = detector.detect(image)
        logger.debug("Found {0} points in {1}s".format(len(points), time.time() - t))
        if len(points) < features_count and sift_peak_threshold > 0.0001:
            sift_peak_threshold = (sift_peak_threshold * 2) / 3
            logger.debug("reducing threshold")
        else:
            logger.debug("done")
            break
    points, desc = descriptor.compute(image, points)

    if desc is not None:
        if config["feature_root"]:
            desc = root_feature(desc)
        points = np.array([(i.pt[0], i.pt[1], i.size, i.angle) for i in points])
    else:
        points = np.array(np.zeros((0, 3)))
        desc = np.array(np.zeros((0, 3)))
    return points, desc
