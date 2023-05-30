import os
import numpy as np
from multiprocessing import Pool


def get_row_index(row, matrix):
    assert len(row.shape) == 1 and len(matrix.shape) == 2 and row.shape[0] == matrix.shape[1]

    for i in range(matrix.shape[0]):
        if np.array_equal(row, matrix[i, :]):
            return i
    return None


def get_rows_index(matrix1, matrix2):
    assert len(matrix1.shape) == 2 and len(matrix2.shape) == 2 and matrix1.shape[1] == matrix2.shape[1]

    with Pool() as p:
        results = [p.apply_async(get_row_index, args=(matrix1[i, :], matrix2)) for i in range(matrix1.shape[0])]
        index = [result.get() for result in results]
    return index


def to_ply(point_cloud, colors, path=os.getcwd(), densify=True):
    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    # out_colors = np.ones_like(out_points) * 255
    verts = np.hstack([out_points, out_colors])

    # cleaning point cloud
    mean = np.mean(verts[:, :3], axis=0)
    temp = verts[:, :3] - mean
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    indx = np.where(dist < np.mean(dist) + 300)
    verts = verts[indx]
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar blue
        property uchar green
        property uchar red
        end_header
        '''
    if not densify:
        with open(path, 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
    else:
        with open(path, 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
