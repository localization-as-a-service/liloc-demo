import time
import zmq
import numpy as np
import cv2
import os
import open3d

from queue import Queue
from collections import deque
from threading import Thread
from zmq.utils.monitor import recv_monitor_message
from utils.depth_camera import DepthCamera, DepthCameraParams
from utils.local_registration import LocalRegistration


class TimestampArrayBuffer:
    def __init__(self, max_buffer_size):
        self.max_buffer_size = max_buffer_size
        self.buffer = deque()

    def add_data(self, timestamp, array):
        if len(self.buffer) >= self.max_buffer_size:
            self.buffer.popleft()
        self.buffer.append((timestamp, array))

    def get_data(self):
        return list(self.buffer)

    def find_nearest(self, target_timestamp):
        nearest_pair = min(self.buffer, key=lambda pair: abs(pair[0] - target_timestamp))
        return nearest_pair


class DepthImageRx(Thread):
    def __init__(self, num_devices: int, queue: Queue):
        super(DepthImageRx, self).__init__()
        self.num_devices = num_devices
        self.queue = queue
        self.buffers = [TimestampArrayBuffer(max_buffer_size=30*5) for _ in range(self.num_devices)]
        self.cameras = []
        self.transformations = []
        
        for i in range(self.num_devices):
            camera_params = DepthCameraParams(f"metadata/device-{i}.json")
            depth_camera = DepthCamera(camera_params)
            transformation = np.loadtxt(f"metadata/device-{i}.txt")

            self.cameras.append(depth_camera)
            self.transformations.append(transformation)
            
        print("Starting Depth Image Rx.")
        

    def run(self):
        while True:
            try:
                depth_image, timestamp, sensor = self.queue.get()
                self.buffers[sensor].add_data(timestamp, depth_image)
                
                time.sleep(0.001)
            except KeyboardInterrupt:
                break
    
    
    def get_global_pcd(self, timestamp):
        targets = []
        for i in range(self.num_devices):
            _, frame = self.buffers[i].find_nearest(timestamp)
            vertices = self.cameras[i].depth_image_to_point_cloud(frame)
            vertices = np.concatenate((vertices, np.ones((vertices.shape[0], 1))), axis=1)
            vertices = np.dot(self.transformations[i], vertices.T).T
            vertices = vertices[:, :3]
            
            targets.append(vertices)
            
        return np.vstack(targets).astype(np.float32)
        

def send_data(socket, timestamp, source, target):
    data = {
        "timestamp": timestamp,
        "source": source.tolist(),
        "target": target.tolist()
    }
    socket.send_json(data, flags=0)
    

def to_timestamp(timestamp):
    return int((timestamp - 116444736000000000) / 10000)
        

def main():
    sequence_dir = "data/simulation/trial_1"
    num_devices = 3
    queue = Queue()
    lr = LocalRegistration()
    
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://127.0.0.1:5557")
    
    rx = DepthImageRx(num_devices, queue)
    rx.start()

    files = os.listdir(sequence_dir)
    files = sorted(files, key=lambda x: int(x.split("_")[0]))
    timestamps = np.array([int(x.split("_")[0]) for x in files])
    delays = np.diff(timestamps) / 1000
    
    global_t = None
    
    
    
    for i, fname in enumerate(files[:-1]):
        file = os.path.join(sequence_dir, fname)
        current_t = int(timestamps[i])
        print("Processing:", file)
        
        if fname.endswith(".png"):
            sensor = int(fname.split("_")[2].split(".")[0])
            
            depth_image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            
            queue.put((depth_image, current_t, sensor))
        else:
            data = np.load(file)
            source = data.get("xyz")
            pose = data.get("pose").astype(np.float32)
            
            lr.update(current_t, source, pose)
            
            if global_t is None:
                global_t = current_t
                
            if current_t - global_t > 1000:
                print(f"Triggering global registration after {current_t - global_t} ms")
                
                target = rx.get_global_pcd(current_t)
                    
                send_data(socket, current_t, source=source, target=target)
                
                global_t = current_t
                
        # time.sleep(delays[i])
        time.sleep(0.005)
        
    x = lr.get_trajectory()
    open3d.visualization.draw_geometries([x])

if __name__ == "__main__":
    main()