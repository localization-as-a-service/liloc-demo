import zmq
import open3d
import numpy as np
import copy
import multiprocessing as mp

import utils.registration as registration
import utils.transform as transform
import utils.pointcloud as pointcloud


class LocalRegistration:
    def __init__(self):
        self.local_t = [np.identity(4)]
        self.global_t = []
        self.pcd_data = []
        self.sequence_ts = []
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect("tcp://localhost:5559")
        
    def _send_data(self, socket, timestamp, pcd, transformation, token):
        data = {
            "timestamp": timestamp,
            "vertices": np.asarray(pcd.points).tolist(),
            "transformation": transformation.tolist(),
            "token": token
        }
        socket.send_json(data, flags=0)
        
    def update(self, timestamp, vertices, pose):
        pcd = pointcloud.make_pcd(vertices)
        pcd = open3d.geometry.voxel_down_sample(pcd, voxel_size=0.05)
        self.pcd_data.append(pcd)
        self.sequence_ts.append(timestamp)
        self.global_t.append(pose)
        
        if len(self.global_t) > 1:
            t = np.dot(transform.inv_transform(self.global_t[-2]), self.global_t[-1])
            self.local_t.append(t)
            
        self._send_data(self.socket, timestamp, pcd, self.local_t[-1], 0)

                
    def get_trajectory(self):
        self._send_data(self.socket, 0, self.pcd_data[-1], np.identity(4), 2)
        self.socket.close()
        
        num_frames = len(self.local_t)
        trajectory_t = [np.identity(4)]

        for t in range(1, num_frames):
            trajectory_t.append(np.dot(trajectory_t[t - 1], self.local_t[t]))
            
        trajectory_pcd = []

        for i in range(num_frames):
            pcd = copy.deepcopy(self.pcd_data[i])
            pcd.transform(trajectory_t[i])
            trajectory_pcd.append(pcd)
            
        return pointcloud.merge_pcds(trajectory_pcd, 0.05)