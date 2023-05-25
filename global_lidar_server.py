import time
import open3d
import zmq
import numpy as np
import cv2
import multiprocessing as mp
from queue import Empty

from typing import List

from utils.depth_camera import DepthCamera, DepthCameraParams

class GlobalLidarHelper(mp.Process):
    def __init__(self, queue_in: mp.Queue, queue_out: mp.Queue, device: int = 0):
        super(GlobalLidarHelper, self).__init__()
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.device = device
        self.camera_params = DepthCameraParams(f"metadata/device-{device}.json")
        self.depth_camera = DepthCamera(self.camera_params)
        self.transformation = np.loadtxt(f"metadata/device-{device}.txt")
        
        print(f"Starting Global Lidar Server for Device {device}.")
    
    def run(self):
        while True:
            try:
                depth_image, timestamp = None, None
                while not self.queue_in.empty():
                    depth_image, timestamp = self.queue_in.get()
                
                if depth_image is None:
                    continue
                
                vertices = self.depth_camera.depth_image_to_point_cloud(depth_image)
                vertices = np.concatenate((vertices, np.ones((vertices.shape[0], 1))), axis=1)
                vertices = np.dot(self.transformation, vertices.T).T
                
                self.queue_out.put((vertices[:, :3], timestamp, self.device))
                
                cv2.imshow("Depth Camera", depth_image)
                key = cv2.waitKey(1)
                # np.savetxt(f"temp/dev_{self.device}.txt", pcd)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
                
            except KeyboardInterrupt:
                break
            except Empty:
                break
            

class GlobalLidarServer(mp.Process):
    def __init__(self, queue: mp.Queue):
        super(GlobalLidarServer, self).__init__()
        self.queue = queue
        self.global_pcds = [np.zeros((1, 3)) for _ in range(4)]
        
    def run(self) -> None:
        while True:
            try:
                vertices, timestamp, device = self.queue.get()
                self.global_pcds[device] = vertices
                print(f"Received Point Cloud from Device {device} at {timestamp}.")
                # global_pcd = np.vstack(self.global_pcds)
                # print(f"Global Point Cloud Shape: {global_pcd.shape}")
                # np.savetxt(f"temp/global.txt", global_pcd)
            except KeyboardInterrupt:
                break


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md["dtype"])
    return A.reshape(md["shape"]), md["timestamp"], md["device"]


def main(queues: List[mp.Queue], gls_processes: List[mp.Process]):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")

    while True:
        try:
            depth_image, timestamp, sensor = recv_array(socket)
            queues[sensor].put((depth_image, timestamp))
            
            # cv2.imshow("Depth Camera", depth_image)
            # key = cv2.waitKey(1)

            # if key & 0xFF == ord('q') or key == 27:
            #     cv2.destroyAllWindows()
            #     break
            time.sleep(0.001)
        except KeyboardInterrupt:
            for p in gls_processes:
                p.terminate()
            break
        
    socket.close()

if __name__ == '__main__':
    num_devices = 3
    queues = [mp.Queue() for _ in range(num_devices)]
    gls_list = []
    global_queue = mp.Queue()
    
    
    for i in range(num_devices):
        gls = GlobalLidarHelper(queues[i], global_queue, i)
        gls.start()
        gls_list.append(gls)
        
        
    merger = GlobalLidarServer(global_queue)
    merger.start()

    main(queues, gls_list)

