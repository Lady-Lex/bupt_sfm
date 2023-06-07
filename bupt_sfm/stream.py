import os
import cv2
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
from itertools import chain

try:
    import rospy
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge, CvBridgeError
    ros_allowed = True
except ModuleNotFoundError as e:
    ros_allowed = False

from .utils import *
from .config import *
from .api import api


def image_stream(cfg, queue):
    """ image generator """
    image_dir = cfg["running_image_dir"]
    calib = cfg["running_calib"]
    stride = cfg["running_stride"]
    skip = cfg["running_skip"]

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

        downscale = cfg["image_downscale"]
        intrinsics[0, 0] = K[0, 0] / float(downscale)
        intrinsics[0, 2] = K[0, 2] / float(downscale)
        intrinsics[1, 1] = K[1, 1] / float(downscale)
        intrinsics[1, 2] = K[1, 2] / float(downscale)
        image = img_downscale(image, downscale)

        h, w, _ = image.shape
        # image = image[:h - h % 16, :w - w % 16]

        queue.put((t, image, intrinsics))
    queue.put((-1, image, intrinsics))


class ros_topic_stream:
    def __init__(self, cfg, queue):
        self.t = 0
        self.K = np.eye(3)
        self.D = np.zeros(5)
        self.info_got = False
        self.bridge = CvBridge()

        self.cfg = cfg
        self.queue = queue
        self.camera_info_topic = self.cfg["running_calib"]
        self.image_topic = self.cfg["running_image_dir"]
        self.stride = self.cfg["running_stride"]
        self.skip = self.cfg["running_skip"]

        self.callback_stopped = False

        rospy.init_node('sfm_image_stream', anonymous=True)
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback,
                                                queue_size=1)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)

        rospy.spin()

    def camera_info_callback(self, data):
        fx = data.K[0]
        fy = data.K[4]
        cx = data.K[2]
        cy = data.K[5]

        self.K[0, 0] = fx
        self.K[0, 2] = cx
        self.K[1, 1] = fy
        self.K[1, 2] = cy

        k1 = data.D[0]
        k2 = data.D[1]
        p1 = data.D[2]
        p2 = data.D[3]
        k3 = data.D[4]

        self.D[0] = k1
        self.D[1] = k2
        self.D[2] = p1
        self.D[3] = p2
        self.D[4] = k3

        self.info_got = True

    def image_callback(self, data):
        if (not self.info_got) and self.callback_stopped:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn(e)

        if api.stop_shoot.value:
            image = cv2.undistort(cv_image, self.K, self.D)
            intrinsics = np.eye(3)

            downscale = self.cfg["image_downscale"]
            intrinsics[0, 0] = self.K[0, 0] / float(downscale)
            intrinsics[0, 2] = self.K[0, 2] / float(downscale)
            intrinsics[1, 1] = self.K[1, 1] / float(downscale)
            intrinsics[1, 2] = self.K[1, 2] / float(downscale)
            image = img_downscale(image, downscale)

            self.queue.put((-1, image, intrinsics))
            self.callback_stopped = True
            api.stop_shoot.value = False
            print("stop shooting!")
            return
        
        if api.shoot_once.value:
            image = cv2.undistort(cv_image, self.K, self.D)
            intrinsics = np.eye(3)

            downscale = self.cfg["image_downscale"]
            intrinsics[0, 0] = self.K[0, 0] / float(downscale)
            intrinsics[0, 2] = self.K[0, 2] / float(downscale)
            intrinsics[1, 1] = self.K[1, 1] / float(downscale)
            intrinsics[1, 2] = self.K[1, 2] / float(downscale)
            image = img_downscale(image, downscale)

            self.queue.put((self.t, image, intrinsics))
            api.shoot_once.value = False
            print("shoot once!")
            self.t += 1



def run_ros_topic_stream(cfg, queue):
    if ros_allowed:
        ros_topic_stream(cfg, queue)
    else:
        raise Exception("ROS not installed!")
