import time
import open3d
import copy
import zmq
import numpy as np
import cv2
import multiprocessing as mp

from threading import Thread
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
                
                # cv2.imshow("Depth Camera", depth_image)
                # key = cv2.waitKey(1)
                # # np.savetxt(f"temp/dev_{self.device}_{time.time_ns()}.txt", vertices)
                # if key & 0xFF == ord('q') or key == 27:
                #     cv2.destroyAllWindows()
                #     break
                
            except KeyboardInterrupt:
                break
            except Empty:
                break
        print(f"Stopping Global Lidar Server for Device {self.device}.")
        

class GPCStitcher(mp.Process):
    def __init__(self, queue: mp.Queue, event: mp.Value):
        super(GPCStitcher, self).__init__()
        self.queue = queue
        self.event = event
        self.global_pcds = [np.zeros((1, 3)) for _ in range(3)]
    
    @staticmethod
    def send_array(socket, array, timestamp, idx, flags=0, copy=True, track=False):
        md = dict(
            dtype=str(array.dtype),
            shape=array.shape,
            timestamp=timestamp,
            idx=idx
        )
        socket.send_json(md, flags | zmq.SNDMORE)
        return socket.send(array, flags, copy=copy, track=track)
        
    def run(self) -> None:
        # counter = 0
        print("Starting Global Point Cloud Stitcher.")
        while True:
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.connect("tcp://localhost:5557")
            
            try:
                vertices, timestamp, device = self.queue.get()
                self.global_pcds[device] = vertices
                print(f"Received Point Cloud from Device {device} at {timestamp}.")
                
                if self.event.value > 0:
                    # timestamp = copy.copy(self.event.value)
                    global_pcd = np.vstack(self.global_pcds).astype(np.float32).copy()
                    GPCStitcher.send_array(socket, global_pcd, timestamp, 1)
                    # Thread(target=GPCStitcher.send_array, args=(socket, global_pcd, timestamp, 1)).start()
                    print(f"Sending Global Point Cloud ({len(global_pcd)}) to FCGF @ {timestamp}")
                    self.event.value = 0
                # if counter % 100 == 0:
                #     global_pcd = np.vstack(self.global_pcds).astype(np.float32).copy()
                #     self._send_array(socket, global_pcd, 1)
                #     # np.savetxt(f"temp/global_{time.time_ns()}.txt", global_pcd)
                # counter += 1
            except KeyboardInterrupt:
                break
            except InterruptedError:
                break
        print("Stopping Global Point Cloud Stitcher.")
        
            
class GPCServer(mp.Process):
    def __init__(self, event: mp.Value, address: str = "*", port: int = 5556):
        super(GPCServer, self).__init__()
        self.url = f"tcp://{address}:{port}"
        self.event = event
        
    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.bind(self.url)
        print(f"Starting Global Point Cloud Server @ {self.url}")
                
        while True:
            try:
                timestamp = socket.recv_string()
                print(f"Global Point Cloud Server Received Message: {timestamp}")
                # if msg == "send":
                    # self.event.value = 1
                self.event.value = int(timestamp)
            except KeyboardInterrupt:
                break
            except InterruptedError:
                break
            
        socket.close()
        print(f"Stopping Global Point Cloud Server @ {self.url}")


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md["dtype"])
    return A.reshape(md["shape"]), md["timestamp"], md["device"]


def main(queues: List[mp.Queue], processes: List[mp.Process]):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5554")

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
            for p in processes:
                p.terminate()
                p.join()
            break
        
    socket.close()

if __name__ == '__main__':
    num_devices = 3
    queues = [mp.Queue() for _ in range(num_devices)]
    processes = []
    global_queue = mp.Queue()
    send_gpc = mp.Value('i', 0)
    
    
    for i in range(num_devices):
        gls = GlobalLidarHelper(queues[i], global_queue, i)
        gls.start()
        processes.append(gls)
        
    stitcher = GPCStitcher(global_queue, send_gpc)
    processes.append(stitcher)
    stitcher.start()
    
    gpc_server = GPCServer(send_gpc)
    processes.append(gpc_server)
    gpc_server.start()

    main(queues, processes)

