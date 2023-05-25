import time
import open3d
import zmq
import numpy as np
import cv2
import multiprocessing as mp

from typing import List

from utils.depth_camera import DepthCamera, DepthCameraParams

class GlobalLidarServer(mp.Process):
    def __init__(self, queue_in: mp.Queue, queue_out: mp.Queue, device: int = 0):
        super(GlobalLidarServer, self).__init__()
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.device = device
        self.camera_params = DepthCameraParams(f"metadata/device-{device}.json")
        self.depth_camera = DepthCamera(self.camera_params)
    
    
    def run(self):
        while True:
            try:
                depth_image, timestamp = self.queue_in.get()
                # pcd = self.depth_camera.depth_image_to_point_cloud(depth_image)
                # self.queue_out.put((pcd, timestamp))
                cv2.imshow("Depth Camera", depth_image)
                key = cv2.waitKey(1)

                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
                
            except KeyboardInterrupt:
                break
        

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md["dtype"])
    return A.reshape(md["shape"]), md["timestamp"], md["device"]


def main(queues: List[mp.Queue]):
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
            
        except KeyboardInterrupt:
            break
        
    socket.close()

if __name__ == '__main__':
    num_devices = 3
    queues = [mp.Queue() for _ in range(num_devices)]
    gls_list = []
    global_queue = mp.Queue()
    
    for i in range(num_devices):
        gls = GlobalLidarServer(queues[i], global_queue, i)
        gls.start()
        gls_list.append(gls)

    
    main(queues)
    
    for gls in gls_list:
        gls.join()

