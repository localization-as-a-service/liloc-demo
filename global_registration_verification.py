import open3d
import numpy as np
import pandas as pd
import os
import glob
import tqdm
import copy
import zmq
import time

import utils.pointcloud as pointcloud
import utils.registration as registration
import utils.grid_search_rs_unopt as grid_search
import utils.transform as transform


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



class GlobalRegistrationVerification:
    
    def __init__(self):
        self.local_t = []
        self.global_t = []
        self.sequence_ts = []
        self.global_inds = []
        self.global_pcds = []
        self.local_pcds = []
        self.global_target_t = []
        self.found_correct_global = False
        self.found_correct_global_at = -1
        
    def update_local(self, local_timestamp, local_pcd, local_transformation):
        self.sequence_ts.append(local_timestamp)
        self.local_t.append(local_transformation)
        self.local_pcds.append(local_pcd)
        
        if self.found_correct_global:
            self.global_t.append(np.dot(self.global_t[-1], local_transformation))
        else:
            self.global_t.append(np.identity(4))
        
    
    def update_global(self, global_timestmap, global_pcd, global_transformation):
        index = np.argwhere(np.array(self.sequence_ts) == global_timestmap).flatten()
        print(index)
        
        if len(index) == 0:
            print(f"Timestamp {global_timestmap} not found in sequence.")
        else:
            index = index[0]
        
        self.global_inds.append(index)
        self.global_target_t.append(global_transformation)
        self.global_pcds.append(global_pcd)
        
        if len(self.global_inds) > 2 and not self.found_correct_global:
            self.verify()
            
    
    def verify(self):
        for t in range(len(self.global_inds)):
            if t > 1:
                print(f"Global registration verification: {t}/{len(self.global_inds)}")
                total = 0
                for i in range(t, t - 3, -1):
                    if np.sum(self.global_target_t[i]) == 4:
                        total += 1
                        
                print(f"Total invalid global registrations: {total}")        
                if total > 1: return
                
                print(f"Validating and correcting global registrations.")
                try:
                    self.global_target_t[t - 2], self.global_target_t[t - 1], self.global_target_t[t] = validate(
                        self.global_target_t[t - 2], self.global_target_t[t - 1], self.global_target_t[t], 
                        merge_transformation_matrices(self.global_inds[t - 2], self.global_inds[t - 1], self.local_t),
                        merge_transformation_matrices(self.global_inds[t - 1], self.global_inds[t], self.local_t),
                        max_rot=2, max_dist=0.1
                    )
                    self.found_correct_global = True
                    self.found_correct_global_at = t
                except Exception as e:
                    print(f"Exception:", e)
                    return
        
        if self.found_correct_global:
            self.global_t[self.global_inds[self.found_correct_global_at]] = self.global_target_t[self.found_correct_global_at]

            for t in range(self.global_inds[self.found_correct_global_at] + 1, len(self.global_t)):
                self.global_t[t] = np.dot(self.global_t[t - 1], self.local_t[t])
                
            for t in range(self.global_inds[self.found_correct_global_at] - 1, -1, -1):
                self.global_t[t] = np.dot(self.global_t[t + 1], transform.inv_transform(self.local_t[t + 1]))
    

def recv_array(socket, flags=0, copy=True, track=False):
    data = socket.recv_json(flags=flags)
    timestamp = data["timestamp"]
    pcd = pointcloud.make_pcd(np.array(data["vertices"]))
    transformation = np.array(data["transformation"])
    token = data["token"]
    return timestamp, pcd, transformation, token


def main():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5559")
    
    grv = GlobalRegistrationVerification()
    
    while True:
        try:
            timestamp, pcd, transformation, token = recv_array(socket)
            print(f"Timestamp: {timestamp} | Token: {token}")
            if token == 0:
                grv.update_local(timestamp, pcd, transformation)
            elif token == 1:
                grv.update_global(timestamp, pcd, transformation)
            else:
                # for i in range(0, len(grv.sequence_ts), 50):
                #     registration.view(grv.local_pcds[i], grv.global_pcds[0], grv.global_t[i])  
                vis = open3d.visualization.Visualizer()
                vis.create_window()
                
                num_frames = len(grv.sequence_ts)

                global_pcd = grv.global_pcds[0]
                global_pcd.paint_uniform_color([0, 0.651, 0.929])

                local_pcd = grv.local_pcds[0]
                local_pcd.transform(grv.global_t[0])
                local_pcd.paint_uniform_color([1, 0.706, 0])

                trajectory = [grv.global_t[i][:3, 3].tolist() for i in range(num_frames) if np.sum(grv.global_t[i]) != 4]
                lines = [[i, i + 1] for i in range(len(trajectory) - 1)]
                colors = [[1, 0, 0] for i in range(len(lines))]

                line_set = open3d.geometry.LineSet()

                vis.add_geometry(global_pcd)
                vis.add_geometry(local_pcd)
                vis.add_geometry(line_set)
                
                skipped_frames = 0

                for i in range(num_frames):
                    if np.sum(grv.global_t[i]) == 4: 
                        skipped_frames += 1
                        continue
                    
                    global_pcd_t = grv.global_pcds[0]
                    global_pcd.points = global_pcd_t.points
                    global_pcd.paint_uniform_color([0, 0.651, 0.929])
                    
                    local_pcd_t = grv.local_pcds[i]
                    local_pcd_t.transform(grv.global_t[i])
                    
                    local_pcd.points = local_pcd_t.points
                    local_pcd.paint_uniform_color([1, 0.706, 0])
                    
                    line_set.points = open3d.utility.Vector3dVector(trajectory[:i+1])
                    line_set.lines = open3d.utility.Vector2iVector(lines[:i - skipped_frames])
                    line_set.colors = open3d.utility.Vector3dVector(colors[:i - skipped_frames])
                    
                    vis.update_geometry()
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.1)
                    
                vis.destroy_window()
                
                break
            
        except KeyboardInterrupt:
            break
        
    socket.close()

if __name__ == "__main__":
    main()