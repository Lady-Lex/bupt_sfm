import cv2
import numpy as np


def triangulationE(P1, P2, points1, points2):
    """三角化"""
    points1 = points1.T
    points2 = points2.T
    points4d = cv2.triangulatePoints(P1, P2, points1, points2).T
    points3d = cv2.convertPointsFromHomogeneous(points4d)[:, 0, :]
    return points3d


def triangulationF():
    pass


def triangulationH():
    pass
