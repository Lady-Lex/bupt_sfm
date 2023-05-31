import os
import cv2
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
from itertools import chain

from .config import *


def image_stream(queue, image_dir, calib, stride, skip=0):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    intrinsics = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(image_dir).glob(e) for e in img_exts))[skip::stride]
    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 1:
            downscale = 2
            intrinsics[0, 0] = K[0, 0] / float(downscale)
            intrinsics[0, 2] = K[0, 2] / float(downscale)
            intrinsics[1, 1] = K[1, 1] / float(downscale)
            intrinsics[1, 2] = K[1, 2] / float(downscale)
            image = img_downscale(image, downscale)
        else:
            intrinsics = K

        h, w, _ = image.shape
        image = image[:h - h % 16, :w - w % 16]

        queue.put((t, image, intrinsics))
    queue.put((-1, image, intrinsics))


def video_stream(queue, imagedir, calib, stride, skip=0):
    """ video generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    cap = cv2.VideoCapture(imagedir)

    t = 0

    for _ in range(skip):
        ret, image = cap.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

        if not ret:
            break

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[:h - h % 16, :w - w % 16]

        intrinsics = np.array([fx * .5, fy * .5, cx * .5, cy * .5])
        queue.put((t, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
    cap.release()


def img_downscale(img, downscale):
    downscale = int(downscale / 2)
    i = 1
    while i <= downscale:
        img = cv2.pyrDown(img)
        i = i + 1
    return img
