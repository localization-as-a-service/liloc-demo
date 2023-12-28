import time
import open3d
import copy
import zmq
import numpy as np
import cv2
import multiprocessing as mp

from utils.pointcloud import make_pcd
from collections import deque
from threading import Thread
from queue import Empty
from zmq.utils.monitor import recv_monitor_message
from typing import List
from threading import Lock
from utils.depth_camera import DepthCamera, DepthCameraParams


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


class GlobalLidarHelper(Thread):
    def __init__(self, queue_in: mp.Queue, device: int = 0):
        super(GlobalLidarHelper, self).__init__()
        self.queue_in = queue_in
        self.buffer = TimestampArrayBuffer(max_buffer_size=30*5)
        self.device = device
        self.camera_params = DepthCameraParams(f"metadata/device-{device}.json")
        self.depth_camera = DepthCamera(self.camera_params)
        self.transformation = np.loadtxt(f"metadata/device-{device}.txt")
        self.lock = Lock()
        self.running = True
        
        print(f"Starting Global Lidar Server for Device {device}.")
    
    def run(self):
        while self.running:
            try:
                depth_image, timestamp = None, None
                while not self.queue_in.empty():
                    depth_image, timestamp = self.queue_in.get()
                
                if depth_image is None:
                    continue
                
                self.lock.acquire()
                
                self.buffer.add_data(timestamp, depth_image)
                
                self.lock.release()
                
                # cv2.imshow(f"Device {self.device + 1}", depth_image)
        
                # key = cv2.waitKey(1)
                
                # cv2.imwrite(f"calib/dev_{self.device}_{timestamp}.png", depth_image)
                
                # if key & 0xFF == ord('q') or key == 27:
                #     break
                
            except KeyboardInterrupt:
                break
            except Empty:
                break
        
        # cv2.destroyAllWindows()
    
        print(f"Stopping Global Lidar Server for Device {self.device}.")
        
    def get_latest_depth_frame(self, timestamp):
        self.lock.acquire()
        _, frame = self.buffer.find_nearest(timestamp)
        self.lock.release()
        vertices = self.depth_camera.depth_image_to_point_cloud(frame)
        vertices = np.concatenate((vertices, np.ones((vertices.shape[0], 1))), axis=1)
        vertices = np.dot(self.transformation, vertices.T).T
        
        return vertices[:, :3]
    
    def stop(self):
        self.running = False
        

class GlobalLiDARController:
    def __init__(self, num_devices: int):
        self.num_devices = num_devices
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5556")
        
        connections = 0
        events_socket = self.socket.get_monitor_socket(events=zmq.EVENT_HANDSHAKE_SUCCEEDED)
        
        while connections < self.num_devices:
            # this will block until a handshake was successful
            recv_monitor_message(events_socket)  
            connections += 1
            print("Connection established to ", connections)
        
    def send(self, message: str):
        self.socket.send_string(message)
            
    def close(self):
        self.socket.close()
        

class DepthImageRx(mp.Process):
    def __init__(self, queues: List[mp.Queue]):
        super(DepthImageRx, self).__init__()
        self.queues = queues
        
    def recv_array(self, socket, flags=0, copy=True, track=False):
        md = socket.recv_json(flags=flags)
        msg = socket.recv(flags=flags, copy=copy, track=track)
        buf = memoryview(msg)
        A = np.frombuffer(buf, dtype=md["dtype"])
        return A.reshape(md["shape"]), md["timestamp"], md["device"]
    
    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind("tcp://*:5554")

        while True:
            try:
                depth_image, timestamp, sensor = self.recv_array(socket)
                self.queues[sensor].put((depth_image, timestamp))
                
                # cv2.imshow(f"Depth Camera {sensor}", depth_image)
                # key = cv2.waitKey(1)

                # if key & 0xFF == ord('q') or key == 27:
                #     cv2.destroyAllWindows()
                #     break
                
                time.sleep(0.001)
            except KeyboardInterrupt:
                break
            
        socket.close()
        

if __name__ == '__main__':
    num_devices = 3
    queues = [mp.Queue() for _ in range(num_devices)]
    helpers: List[GlobalLidarHelper] = []
        
    for i in range(num_devices):
        gls = GlobalLidarHelper(queues[i], i)
        gls.start()
        helpers.append(gls)
        
    rx = DepthImageRx(queues)
    rx.start()
    
    controller = GlobalLiDARController(num_devices)
    time.sleep(5)
    controller.send("global_lidar start")
    time.sleep(8)
    controller.send("global_lidar stop")
    
    current_t = time.time() * 1000
    
    global_pcd = open3d.geometry.PointCloud()
    static_pcd = open3d.io.read_point_cloud("../calibration/env_static.pcd")

    
    for helper in helpers:
        pcd = helper.get_latest_depth_frame(current_t - 1000)
        pcd = make_pcd(pcd)
        global_pcd += pcd
        helper.stop()

    global_pcd += static_pcd

    open3d.visualization.draw_geometries([global_pcd])
        
    controller.close()
    
    
    
    

