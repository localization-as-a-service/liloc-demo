import open3d
import numpy as np
import os

import utils.registration as registration
import utils.FCGF as FCGF
import utils.transform as transform
import utils.pointcloud as pointcloud

from concurrent.futures import ThreadPoolExecutor


# def get_limits(pcd):
#     pcd_points = np.asarray(pcd.points)

#     x_min, y_min, z_min = np.min(pcd_points, axis=0)
#     x_max, y_max, z_max = np.max(pcd_points, axis=0)

#     return x_min, x_max, y_min, y_max, z_min, z_max


# def get_grid(pcd, cell_size):
#     x_min, x_max, y_min, y_max, z_min, z_max = get_limits(pcd)
#     y_val = np.mean([y_min, y_max])

#     points = []
#     x_n = int((x_max - x_min) // cell_size)
#     z_n = int((z_max - z_min) // cell_size)
#     for i in range(z_n):
#         z0 = float(z_min + cell_size * (i + 1))
#         for j in range(x_n):
#             x0 = float(x_min + cell_size * (j + 1))
#             points.append([x0, y_val, z0])

#     return points

# def filter_indices(points, p, cell_size):
#     px_min = p[0] - cell_size
#     px_max = p[0] + cell_size
#     pz_min = p[2] - cell_size
#     pz_max = p[2] + cell_size
#     xf = np.logical_and(points[:, 0] > px_min, points[:, 0] < px_max)
#     zf = np.logical_and(points[:, 2] > pz_min, points[:, 2] < pz_max)
#     return np.logical_and(xf, zf)


def get_limits(xyz):
    x_min, y_min, z_min = np.min(xyz, axis=0)
    x_max, y_max, z_max = np.max(xyz, axis=0)

    return x_min, x_max, y_min, y_max, z_min, z_max


def get_grid(xyz, cell_size):
    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(xyz)
    z0 = np.mean([z_min, z_max])

    points = []
    x_n = int((x_max - x_min) // cell_size)
    y_n = int((y_max - y_min) // cell_size)
    for i in range(y_n):
        y0 = float(y_min + cell_size * (i + 1))
        for j in range(x_n):
            x0 = float(x_min + cell_size * (j + 1))
            points.append([x0, y0, z0])

    return points


def filter_indices(points, p, cell_size):
    px_min = p[0] - cell_size
    px_max = p[0] + cell_size
    py_min = p[1] - cell_size
    py_max = p[1] + cell_size
    xf = np.logical_and(points[:, 0] > px_min, points[:, 0] < px_max)
    yf = np.logical_and(points[:, 1] > py_min, points[:, 1] < py_max)
    return np.logical_and(xf, yf)


def get_cell_features(feature_file, p, cell_size):
    data = np.load(feature_file)
    scores = data["scores"]

    f = filter_indices(data["keypts"], p, cell_size)
    f = np.where(f)[0]

    features = open3d.registration.Feature()
    features.data = data["features"][f].T

    keypts = pointcloud.make_pcd(data["keypts"][f])
    return keypts, features, scores


def register_cell(source, target, source_feat, target_feat, n_ransac, threshold):
    if len(target.points) < 2000:
        return None
    
    result_ransac = registration.exec_ransac(source, target, source_feat, target_feat, n_ransac=n_ransac, threshold=threshold)
    
    return result_ransac


def global_registration(src_feature_file, tgt_feature_file, voxel_size, cell_size, refine_enabled=False):
    global_pcd = FCGF.get_features(tgt_feature_file, voxel_size, pcd_only=True)
    
    center_pts = get_grid(global_pcd, cell_size)
    
    src_feature_file = os.path.join(src_feature_file)
    source, source_feat, _ = FCGF.get_features(src_feature_file, voxel_size)
    
    targets = []
    target_feats = []
    
    delete_indices = []
    
    for i in range(len(center_pts)):
        target, target_feat, _ = get_cell_features(tgt_feature_file, center_pts[i], cell_size)
        
        if len(target.points) < 2000:
            delete_indices.append(i)
            continue
        
        targets.append(target)
        target_feats.append(target_feat)
        
    center_pts = np.delete(center_pts, delete_indices, axis=0)

    reg_result = None
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = []
        for i in range(len(center_pts)):
            results.append(executor.submit(register_cell, source, targets[i], source_feat, target_feats[i], 3, 0.05))
            
        for i in range(len(center_pts)):
            result_ransac = results[i].result()
            
            if not result_ransac: continue
            
            if reg_result is None or (len(reg_result.correspondence_set) < len(result_ransac.correspondence_set) and reg_result.fitness < result_ransac.fitness):
                reg_result = result_ransac
    
    if refine_enabled and reg_result is not None:
        reg_result = registration.exec_icp(source, global_pcd, threshold=0.05, trans_init=reg_result.transformation, max_iteration=200, p2p=False)
    
    return source, global_pcd, reg_result



def validate(T1, T2, T3, t1, t2, max_dist, max_rot):
    c1 = transform.check(T3, np.dot(T2, t2), max_t=max_dist, max_r=max_rot)
    c2 = transform.check(T3, np.dot(np.dot(T1, t1), t2), max_t=max_dist, max_r=max_rot)
    c3 = transform.check(T2, np.dot(T1, t1), max_t=max_dist, max_r=max_rot)

    print(f"Check 1: {c1}, Check 2: {c2}, Check 3: {c3}")
    
    # If two checks are true, the combination is wrong
    if (c1 + c2 + c3) == 2:
        raise Exception("Invalid combination")

    # If two checks are true, the combination is wrong
    if (c1 + c2 + c3) == 0:
        raise Exception("Invalid transformations")

    # If all the checks are valid, there is no need of correction
    if c1 and c2 and c3:
        print(":: No need of correction.")
        return T1, T2, T3
    
    # If two checks are wrong, only one transformation needs correction
    if c1:
        # print(":: Correcting Previous Transformation")
        T1 = np.dot(T2, transform.inv_transform(t1))
    elif c2:
        # print(":: Correcting Current Transformation")
        T2 = np.dot(T1, t1)
    else:
        # print(":: Correcting Future Transformation")
        T3 = np.dot(T2, t2)

    return T1, T2, T3


def merge_transformation_matrices(start_t, end_t, local_t):
    local_ts = np.identity(4)

    for t in range(start_t, end_t):
        local_ts = np.dot(local_t[t + 1], local_ts)
        
    return local_ts


