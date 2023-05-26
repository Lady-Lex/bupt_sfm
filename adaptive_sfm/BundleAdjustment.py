import cv2
import numpy as np
from scipy.optimize import least_squares


def ReprojectionError(cloud, poses, track, K):
    i = 0
    repr_error = 0
    while (i < len(poses)):
        Rt = poses[i].reshape((3, 4))
        R = Rt[:3, :3]
        t = Rt[:3, 3]
        r, _ = cv2.Rodrigues(R)

        p = track[:, i:i + 2]
        p_reproj, _ = cv2.projectPoints(cloud, r, t, K, distCoeffs=None)
        p_reproj = p_reproj[:, 0, :]
        # print(p[0], p_reproj[0])
        total_error = cv2.norm(p, p_reproj, cv2.NORM_L2)
        repr_error = repr_error + total_error / len(p)
        i = i + 1
    print(p[0], p_reproj[1])
    return repr_error


def OptimReprojectionError(x, cloud_len, poses_len, track, img_tot):
    tracks_len = track.ravel().shape[0]
    K = x[0:9].reshape((3, 3))
    poses = x[9:9 + poses_len].reshape((img_tot, 12))
    cloud = x[9 + poses_len: 9 + poses_len + cloud_len].reshape((int(cloud_len / 3), 3))
    temp = 9 + poses_len + cloud_len
    tracks = x[temp: temp + tracks_len].reshape((int(cloud_len / 3), 2 * img_tot))

    error = []
    i = 0
    while (i < img_tot):
        Rt = poses[i].reshape((3, 4))
        R = Rt[:3, :3]
        t = Rt[:3, 3]
        r, _ = cv2.Rodrigues(R)
        p = track[:, i:i + 2]
        i = i + 1
        p_reproj, _ = cv2.projectPoints(cloud, r, t, K, distCoeffs=None)
        p_reproj = p_reproj[:, 0, :]
        # print(p[0], p_reproj[0])
        for idx in range(len(p)):
            img_pt = p[idx]
            reprojected_pt = p_reproj[idx]
            # er = (img_pt - reprojected_pt)**2
            er = np.sqrt((img_pt[0] - reprojected_pt[0]) ** 2 + (img_pt[1] - reprojected_pt[1]) ** 2)
            error = error + [er]
    print(p[1], p_reproj[1])
    err_arr = np.array(error).ravel() / len(error)
    # print(np.sum(err_arr))
    return err_arr


def BundleAdjustment(cloud, poses, tracks, K, img_tot):
    # print(cloud.shape, poses.shape, tracks.shape)
    cloud_len = cloud.ravel().shape[0]
    poses_len = poses.ravel().shape[0]
    tracks_len = tracks.ravel().shape[0]
    opt_variables = np.hstack((K.ravel(), poses.ravel()))
    opt_variables = np.hstack((opt_variables, cloud.ravel()))
    opt_variables = np.hstack((opt_variables, tracks.ravel()))
    error_arr = OptimReprojectionError(opt_variables, cloud_len, poses_len, tracks_len, img_tot)
    corrected_values = least_squares(fun=OptimReprojectionError, x0=opt_variables, gtol=2,
                                     args=(cloud_len, poses_len, tracks_len, img_tot))
    corrected_values = corrected_values.x
    K = corrected_values[0:9].reshape((3, 3))
    poses = corrected_values[9:9 + poses_len].reshape((img_tot, 12))
    cloud = corrected_values[9 + poses_len: 9 + poses_len + cloud_len].reshape((int(cloud_len / 3), 3))
    temp = 9 + poses_len + cloud_len
    tracks = corrected_values[temp: temp + tracks_len].reshape((int(cloud_len / 3), 2 * img_tot))
    # print(poses.shape, cloud.shape, tracks.shape, K.shape)
    return cloud, poses, tracks