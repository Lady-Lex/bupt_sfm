import cv2
import time
import logging
import numpy as np
from typing import Any, List, Tuple, Dict, Optional

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


def extract_features(
        image: np.ndarray, config: Dict[str, Any], is_panorama: bool
) -> FeaturesData:
    """Detect features in a color or gray-scale image.

    The type of feature detected is determined by the ``feature_type``
    config option.

    The coordinates of the detected points are returned in normalized
    image coordinates.

    Parameters:
        - image: a color image with shape (h, w, 3) or
                 gray-scale image with (h, w) or (h, w, 1)
        - config: the configuration structure
        - is_panorama : if True, alternate settings are used for feature count and extraction size.

    Returns:
        tuple:
        - points: ``x``, ``y``, ``size`` and ``angle`` for each feature
        - descriptors: the descriptor of each feature
        - colors: the color of the center of each feature
    """
    extraction_size = (
        config["feature_process_size_panorama"]
        if is_panorama
        else config["feature_process_size"]
    )
    features_count = (
        config["feature_min_frames_panorama"]
        if is_panorama
        else config["feature_min_frames"]
    )

    assert len(image.shape) == 3 or len(image.shape) == 2
    image = resized_image(image, extraction_size)
    if len(image.shape) == 2:  # convert (h, w) to (h, w, 1)
        image = np.expand_dims(image, axis=2)
    # convert color to gray-scale if necessary
    if image.shape[2] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image
    feature_type = config["feature_type"].upper()
    points, desc = extract_features_sift(image_gray, config, features_count)

    xs = points[:, 0].round().astype(int)
    ys = points[:, 1].round().astype(int)
    colors = image[ys, xs]
    if image.shape[2] == 1:
        colors = np.repeat(colors, 3).reshape((-1, 3))

    # return points, desc, colors
    return FeaturesData(points, desc if desc is not None else None, colors)


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


class SIFTmatcher:
    config: Dict[str, Any]
    matcher = cv2.BFMatcher()

    @classmethod
    def match_by_images(cls, image1: np.ndarray, image2: np.ndarray) -> Tuple[FeaturesData, FeaturesData]:
        feature1 = extract_features(image1, cls.config, False)
        feature2 = extract_features(image2, cls.config, False)
        return cls.match_by_features(feature1, feature2)

    @classmethod
    def match_by_features(cls, feature1: FeaturesData, feature2: FeaturesData) -> Tuple[FeaturesData, FeaturesData]:
        matches = cls.matcher.knnMatch(feature1.descriptors, feature2.descriptors, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        point0 = np.float32([feature1.points[m.queryIdx].pt for m in good])
        point1 = np.float32([feature2.points[m.trainIdx].pt for m in good])
        descriptor0 = np.float32([feature1.descriptors[m.queryIdx] for m in good])
        descriptor1 = np.float32([feature2.descriptors[m.trainIdx] for m in good])
        color0 = np.float32([feature1.colors[m.queryIdx] for m in good])
        color1 = np.float32([feature2.colors[m.trainIdx] for m in good])

        return FeaturesData(point0, descriptor0, color0), FeaturesData(point1, descriptor1, color1)

    @classmethod
    def search_by_Initialization(cls, feature1: FeaturesData, feature2: FeaturesData, K: np.ndarray) \
            -> Tuple[FeaturesData, FeaturesData, np.ndarray]:
        feature1, feature2 = cls.match_by_features(feature1, feature2)
        # 或者计算基础矩阵
        E, mask = cv2.findEssentialMat(feature1.points, feature2.points, K, method=cv2.RANSAC, prob=0.999, threshold=1,
                                       mask=None)
        feature1.mask(mask)
        feature2.mask(mask)
        return feature1, feature2, E
