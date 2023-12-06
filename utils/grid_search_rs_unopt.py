import os
import open3d
import numpy as np

import utils.registration as registration

from concurrent.futures import ThreadPoolExecutor


def make_pcd(points):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    return pcd


# def get_limits(pcd):
#     x_min, y_min, z_min = np.min(pcd, axis=0)
#     x_max, y_max, z_max = np.max(pcd, axis=0)

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


def make_reg_features(feats):
    features = open3d.registration.Feature()
    features.data = feats.T
    return features


def get_features(vertices, feats, n_random):
    n_keypts = len(vertices)
    indices = np.random.randint(0, n_keypts, int(n_keypts * n_random))
    
    keypts = open3d.geometry.PointCloud()
    keypts.points = open3d.utility.Vector3dVector(vertices[indices])
    
    open3d.geometry.estimate_normals(keypts)
    
    features = open3d.registration.Feature()
    features.data = feats[indices].T
    return keypts, features


def get_cell_features(vertices, feats, p, cell_size, n_random):
    indices = filter_indices(vertices, p, cell_size)
    indices = np.where(indices)[0]
    
    if len(indices) < n_random:
        return None, None
    
    indices = np.random.choice(indices, int(len(indices) * n_random), replace=False)

    features = open3d.registration.Feature()
    features.data = feats[indices].T

    keypts = make_pcd(vertices[indices])
    return keypts, features


def register_cell(source, target, source_feat, target_feat, n_ransac, threshold):
    if len(target.points) < 2000:
        return None
    
    result_ransac = registration.exec_ransac(source, target, source_feat, target_feat, n_ransac=n_ransac, threshold=threshold)
    
    return result_ransac


def global_registration(source, source_feat, global_pcd, global_feat, cell_size, n_random=0.5, refine_enabled=False):
    source, source_feat = get_features(source, source_feat, n_random)
    center_pts = get_grid(global_pcd, cell_size)
    
    targets = []
    target_feats = []
    
    delete_indices = []
    
    for i in range(len(center_pts)):
        target, target_feat = get_cell_features(global_pcd, global_feat, center_pts[i], cell_size, n_random)
        
        if not target or len(target.points) < 2000:
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
    
    global_pcd = make_pcd(global_pcd)
    
    if refine_enabled and reg_result is not None:
        reg_result = registration.exec_icp(source, global_pcd, threshold=0.05, trans_init=reg_result.transformation, max_iteration=200)
    
    return source, global_pcd, reg_result

