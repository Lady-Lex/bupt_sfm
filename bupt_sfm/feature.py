import cv2
import time
import numpy as np
from typing import Any, List, Tuple, Dict, Optional

from bupt_sfm.config import load_config
from bupt_sfm.context import *


class FeaturesData:
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


class MatchedFeatures(FeaturesData):
    def __init__(self, points: np.ndarray, descriptors: Optional[np.ndarray], colors: np.ndarray, index: List[int]):
        super().__init__(points, descriptors, colors)
        self.index = index

    def mask(self, mask: np.ndarray) -> "MatchedFeatures":
        return MatchedFeatures(
            self.points[mask],
            self.descriptors[mask] if self.descriptors is not None else None,
            self.colors[mask],
            [self.index[i] for i in np.where(mask)[0]]
        )


def resized_image(image: np.ndarray, max_size: int) -> np.ndarray:
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
    points, desc = extract_features_sift(image_gray, config, features_count)

    xs = points[:, 0].round().astype(int)
    ys = points[:, 1].round().astype(int)
    colors = image[ys, xs]
    if image.shape[2] == 1:
        colors = np.repeat(colors, 3).reshape((-1, 3))

    return FeaturesData(points, desc if desc is not None else None, colors)


def extract_features_sift(
        image: np.ndarray, config: Dict[str, Any], features_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    sift_edge_threshold = config["sift_edge_threshold"]
    sift_peak_threshold = float(config["sift_peak_threshold"])
    # SIFT support is in cv2 main from version 4.4.0
    if OPENCV44 or OPENCV5:
        # OpenCV versions concerned /** 3.4.11, >= 4.4.0 **/  ==> Sift became free since March 2020
        detector = cv2.SIFT_create(
            edgeThreshold=sift_edge_threshold, contrastThreshold=sift_peak_threshold
        )
        descriptor = detector
    elif OPENCV3 or OPENCV4:
        try:
            # OpenCV versions concerned /** 3.2.x, 3.3.x, 3.4.0, 3.4.1, 3.4.2, 3.4.10, 4.3.0, 4.4.0 **/
            detector = cv2.xfeatures2d.SIFT_create(
                edgeThreshold=sift_edge_threshold, contrastThreshold=sift_peak_threshold
            )
        except AttributeError as ae:
            # OpenCV versions concerned /** 3.4.3, 3.4.4, 3.4.5, 3.4.6, 3.4.7, 3.4.8, 3.4.9, 4.0.x, 4.1.x, 4.2.x **/
            raise ae
        descriptor = detector
    else:
        detector = cv2.FeatureDetector_create("SIFT")
        descriptor = cv2.DescriptorExtractor_create("SIFT")
        detector.setDouble("edgeThreshold", sift_edge_threshold)
    while True:
        t = time.time()
        # SIFT support is in cv2 main from version 4.4.0
        if OPENCV44 or OPENCV5:
            detector = cv2.SIFT_create(
                edgeThreshold=sift_edge_threshold, contrastThreshold=sift_peak_threshold
            )
        elif OPENCV3:
            detector = cv2.xfeatures2d.SIFT_create(
                edgeThreshold=sift_edge_threshold, contrastThreshold=sift_peak_threshold
            )
        else:
            detector.setDouble("contrastThreshold", sift_peak_threshold)
        points = detector.detect(image)
        if len(points) < features_count and sift_peak_threshold > 0.0001:
            sift_peak_threshold = (sift_peak_threshold * 2) / 3
        else:
            break

    points, desc = descriptor.compute(image, points)

    if desc is not None:
        if config["feature_root"]:
            desc = root_feature(desc)
        # points = np.array([(i.pt[0], i.pt[1], i.size, i.angle) for i in points])
        points = np.array([(i.pt[0], i.pt[1]) for i in points])
    else:
        points = np.array(np.zeros((0, 3)))
        desc = np.array(np.zeros((0, 3)))
    return points, desc


class SIFTmatcher:
    matcher = cv2.BFMatcher()

    @classmethod
    def match_by_images(cls, cfg: Dict[str, Any], image1: np.ndarray, image2: np.ndarray) -> Tuple[MatchedFeatures, MatchedFeatures]:
        feature1 = extract_features(image1, cfg, False)
        feature2 = extract_features(image2, cfg, False)
        return cls.match_by_features(cfg, feature1, feature2)

    @classmethod
    def match_by_features(cls, cfg: Dict[str, Any], feature1: FeaturesData, feature2: FeaturesData) -> Tuple[MatchedFeatures, MatchedFeatures]:
        matches = cls.matcher.knnMatch(feature1.descriptors, feature2.descriptors, k=2)
        good = []
        for m, n in matches:
            if m.distance < cfg["feature_distance_ratio"] * n.distance:
                good.append(m)

        point0 = np.float32([feature1.points[m.queryIdx] for m in good])
        point1 = np.float32([feature2.points[m.trainIdx] for m in good])
        descriptor0 = np.float32([feature1.descriptors[m.queryIdx] for m in good])
        descriptor1 = np.float32([feature2.descriptors[m.trainIdx] for m in good])
        color0 = np.float32([feature1.colors[m.queryIdx] for m in good])
        color1 = np.float32([feature2.colors[m.trainIdx] for m in good])

        return MatchedFeatures(point0, descriptor0, color0, [m.queryIdx for m in good]), \
            MatchedFeatures(point1, descriptor1, color1, [m.trainIdx for m in good])

    @classmethod
    def search_by_Initialization(cls, cfg: Dict[str, Any], feature1: FeaturesData, feature2: FeaturesData, K: np.ndarray) \
            -> Tuple[MatchedFeatures, MatchedFeatures, np.ndarray]:
        matched_feature1, matched_feature2 = cls.match_by_features(cfg, feature1, feature2)
        E, mask = cv2.findEssentialMat(matched_feature1.points, matched_feature2.points, K, method=cv2.RANSAC,
                                       prob=0.999, threshold=1,
                                       mask=None)
        if mask is None:
            pass
        else:
            matched_feature1.mask(mask.ravel() == 1)
            matched_feature2.mask(mask.ravel() == 1)
        return matched_feature1, matched_feature2, E
