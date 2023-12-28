import open3d
import numpy as np
import pandas as pd
import os
import glob
import tqdm
import copy
import zmq
import time

import matplotlib.pyplot as plt

import utils.pointcloud as pointcloud
import utils.registration as registration
import utils.grid_search_rs_unopt as grid_search
import utils.transform as transform
import multiprocessing as mp

from utils.verification import GlobalRegistrationVerification



def main():
    grv = GlobalRegistrationVerification()
    
    local_pcd = None
    global_pcd = None
    line_set = None
    trajectory_t = np.identity(4)
    
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    
    while True:
        try:
            try:
                timestamp, pcd, transformation, token = recv_array(socket)
                print(f"Timestamp: {timestamp} | Token: {token}")
            except zmq.error.Again:
                vis.update_geometry()
                vis.poll_events()
                vis.update_renderer()
                continue
            
            if token == 0:
                grv.update_local(timestamp, pcd, transformation)
                
                if not grv.found_correct_global:
                    if not local_pcd:
                        local_pcd = copy.deepcopy(pcd)
                        local_pcd = pointcloud.remove_outliers(local_pcd)
                        local_pcd.transform(transformation)
                        local_pcd.paint_uniform_color([1, 0.706, 0])
                        vis.add_geometry(local_pcd)
                    else:
                        temp_pcd = copy.deepcopy(pcd)
                        temp_pcd = pointcloud.remove_outliers(temp_pcd)
                        trajectory_t = np.dot(trajectory_t, transformation)
                        temp_pcd.transform(trajectory_t)
                        local_pcd += temp_pcd
                        local_pcd.paint_uniform_color([1, 0.706, 0])
                else:
                    trajectory = np.array([grv.global_t[i][:3, 3] for i in range(len(grv.global_t)) if np.sum(grv.global_t[i]) != 4])
                    lines = [[i, i + 1] for i in range(len(trajectory) - 1)]
                    colors = [[1, 0, 0] for _ in range(len(lines))]
                
                    line_set.points = open3d.utility.Vector3dVector(trajectory)
                    line_set.lines = open3d.utility.Vector2iVector(lines)
                    line_set.colors = open3d.utility.Vector3dVector(colors)
                    
                vis.update_geometry()
                vis.poll_events()
                vis.update_renderer()
                
                time.sleep(0.02)
            elif token == 1:
                grv.update_global(timestamp, pcd, transformation)
                
                if grv.found_correct_global:
                    stop_event.value = 1
                    
                    if not global_pcd:
                        vis.remove_geometry(local_pcd)

                        global_pcd = copy.deepcopy(pcd)
                        global_pcd.paint_uniform_color([0, 0.651, 0.929])
                        vis.add_geometry(global_pcd)
                    
                        line_set = open3d.geometry.LineSet()
                        
                        trajectory = np.array([grv.global_t[i][:3, 3] for i in range(len(grv.global_t)) if np.sum(grv.global_t[i]) != 4])
                        lines = [[i, i + 1] for i in range(len(trajectory) - 1)]
                        colors = [[1, 0, 0] for _ in range(len(lines))]
                
                        line_set.points = open3d.utility.Vector3dVector(trajectory)
                        line_set.lines = open3d.utility.Vector2iVector(lines)
                        line_set.colors = open3d.utility.Vector3dVector(colors)
                        
                        vis.add_geometry(line_set)
                    else:
                        global_pcd.points = copy.deepcopy(pcd).points
                        global_pcd.paint_uniform_color([0, 0.651, 0.929])
                        
                        trajectory = np.array([grv.global_t[i][:3, 3] for i in range(len(grv.global_t)) if np.sum(grv.global_t[i]) != 4])
                        lines = [[i, i + 1] for i in range(len(trajectory) - 1)]
                        colors = [[1, 0, 0] for _ in range(len(lines))]
                
                        line_set.points = open3d.utility.Vector3dVector(trajectory)
                        line_set.lines = open3d.utility.Vector2iVector(lines)
                        line_set.colors = open3d.utility.Vector3dVector(colors)
                    
                    vis.update_geometry()
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.02)                
        except KeyboardInterrupt:
            vis.destroy_window()
            break
        
    socket.close()
    if global_registration.is_alive():
        global_registration.terminate()
        
    global_registration.join()

if __name__ == "__main__":
    main()