import cv2
import numpy as np


def feat_to_tracks(kp, hs):
    # print(kp.shape, poses.shape)
    tot_corrs = hs.shape[0]
    i = 0
    track_pts = np.array(kp)
    while (i < tot_corrs):
        H = hs[i].reshape(3, 3)
        # print(H)
        # print(kp[0])
        kp_h = cv2.convertPointsToHomogeneous(kp)[:, 0, :]
        Hinv = np.linalg.inv(H)
        kp_h = np.array([np.matmul(Hinv, kp_) for kp_ in kp_h])
        kp = cv2.convertPointsFromHomogeneous(kp_h)[:, 0, :]
        track_pts = np.hstack((kp, track_pts))

        i = i + 1
    return track_pts