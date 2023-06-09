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
        self.imu_data = np.array([]).reshape(0, 7)
        self.filtered_imu_data = np.array([]).reshape(0, 7)
        self.local_t = [np.identity(4)]
        self.velocity = np.zeros(3)
        self.pcd_data = []
        self.sequence_ts = []
        self.calibrated = False
        self.gravity = None
        self.window_len = 1600
        self.dt = 0.0025
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
        
    def update(self, data, timestamp):
        if data.ndim == 1:
            self.imu_data = np.concatenate([self.imu_data, data.reshape(1, -1)], axis=0)
            if self.calibrated:
                # removing gravity
                self.imu_data[-1, 1:4] = self.imu_data[-1, 1:4] - self.gravity
                # moving average filter
                avg = np.mean(self.imu_data[-self.window_len:, 1:4], axis=0)
                imu_data_copy = self.imu_data[-1].copy()
                imu_data_copy[1:4] = imu_data_copy[1:4] - avg
                self.filtered_imu_data = np.concatenate([self.filtered_imu_data, imu_data_copy.reshape(1, -1)], axis=0)
        else:
            pcd = pointcloud.make_pcd(data)
            pcd = open3d.voxel_down_sample(pcd, voxel_size=0.05)
            self.pcd_data.append(pcd)
            self.sequence_ts.append(timestamp)
            
            if not self.calibrated:
                # calculate gravity vector
                self.imu_data = np.array(self.imu_data)
                self.gravity = np.mean(self.imu_data, axis=0)[1:4]
                self.imu_data[:, 1:4] = self.imu_data[:, 1:4] - self.gravity
                self.calibrated = True
                # send first pointcloud
                self._send_data(self.socket, timestamp, pcd, np.identity(4), 0)
            else:
                if len(self.filtered_imu_data) == 0:
                    print("No IMU data available")
                    del self.pcd_data[-1]
                    del self.sequence_ts[-1]
                    return
                
                rotation_matrix = np.identity(4)
                translation = np.zeros(3)

                for i in range(len(self.filtered_imu_data[self.filtered_imu_data[:, 0] <= timestamp])):
                    v = self.filtered_imu_data[i]
                    
                    # current displacement and rotation
                    da = np.degrees([v[j + 4] * self.dt for j in range(3)])
                    
                    acceleration = v[1:4]

                    d = [(self.velocity[j] * self.dt) + (0.5 * acceleration[j] * self.dt * self.dt) for j in range(3)]
                    d = np.dot(rotation_matrix, np.array([*d, 1]))
                    
                    translation = translation + d[:3]
                    self.velocity = [self.velocity[j] + acceleration[j] * self.dt for j in range(3)]
                    
                    rotation_matrix = transform.rotate_transformation_matrix(rotation_matrix, da[0], da[1], da[2])
                    
                trans_mat = np.identity(4)
                trans_mat[:3, 3] = translation
                trans_mat[:3, :3] = rotation_matrix[:3, :3]

                source = copy.deepcopy(self.pcd_data[-1])
                target = copy.deepcopy(self.pcd_data[-2])

                refined_transform = registration.exec_icp(source, target, 0.05, trans_mat, 200)
                registration.describe(source, target, refined_transform)
                # registration.view(source, target, refined_transform.transformation)
                refined_transform = np.asarray(refined_transform.transformation)
                
                # velocity = refined_transform[:3, 3] * 1e3 / (filtered_imu_data[-1, 0] - filtered_imu_data[0, 0])
                self.local_t.append(refined_transform)
                # send subsequent pointclouds
                self._send_data(self.socket, timestamp, pcd, refined_transform, 0)

                # self.filtered_imu_data = np.array([]).reshape(0, 7)
                self.filtered_imu_data = self.filtered_imu_data[self.filtered_imu_data[:, 0] > timestamp]
                
    def get_trajectory(self):
        # local_pcds = [np.asarray(self.pcd_data[i].points) for i in range(len(self.pcd_data))]
        # np.savez_compressed("temp/trajectory/local.npz", sequence_ts=self.sequence_ts, local_t=self.local_t, local_pcds=np.array(local_pcds, dtype=object))
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


class SecondaryDevice(mp.Process):
    def __init__(self, queue: mp.Queue):
        super(SecondaryDevice, self).__init__()
        self.queue = queue
        
    def run(self):
        lr = LocalRegistration()
        while True:
            try:
                data, timestamp = self.queue.get()
                
                if data is None:
                    x = lr.get_trajectory()
                    open3d.visualization.draw_geometries([x])
                    break
                
                lr.update(data, timestamp)
            except KeyboardInterrupt:
                break


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    array = np.frombuffer(buf, dtype=md["dtype"])
    return array.reshape(md["shape"]), md["sensor"], md["timestamp"]


def main():
    queue = mp.Queue()
    secondary_device = SecondaryDevice(queue)
    secondary_device.start()

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")

    while True:
        try:
            data, sensor, timestamp = recv_array(socket)
            # print(f"Sensor: {sensor} | Data: {data.shape}")
            queue.put((data, timestamp))
            # socket.send(b"OK")
            
            if sensor == 2:
                queue.put((None, None))
                break
        except KeyboardInterrupt:
            break
        
    socket.close()
    
if __name__ == "__main__":
    main()