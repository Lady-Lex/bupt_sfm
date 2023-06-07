import cv2


# Handle different OpenCV versions
OPENCV5: bool = int(cv2.__version__.split(".")[0]) >= 5
OPENCV4: bool = int(cv2.__version__.split(".")[0]) >= 4
OPENCV44: bool = (
    int(cv2.__version__.split(".")[0]) == 4 and int(cv2.__version__.split(".")[1]) >= 4
)
OPENCV3: bool = int(cv2.__version__.split(".")[0]) >= 3