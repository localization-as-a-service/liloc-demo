import sys

sys.path.append('./hl2ss/Comp_Engine')

import hl2ss_imshow
import hl2ss
import hl2ss_3dcv

import time
import zmq
import numpy as np
import cv2
import os

from threading import Thread
from zmq.utils.monitor import recv_monitor_message


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
        

class DepthImageRx(Thread):
    def __init__(self, num_devices: int, output_dir: str):
        super(DepthImageRx, self).__init__()
        self.num_devices = num_devices
        self.output_dir = output_dir
        
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
                cv2.imwrite(f"{self.output_dir}/{timestamp}_device_{sensor}.png", depth_image)
                
                time.sleep(0.001)
            except KeyboardInterrupt:
                break
            
        socket.close()
        

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
    # HoloLens address
    host = "192.168.10.149"
    calibration_path = './hl2ss/calibration'
    output_dir = 'data/simulation/trial_5'
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_devices = 3
    utc_offset = 133441564321791243 # client_rc.py to get offset
    
    rx = DepthImageRx(num_devices, output_dir)
    rx.start()
    
    controller = GlobalLiDARController(num_devices)
    
    try:
        time.sleep(5)
        controller.send("global_lidar start")
        
        calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)

        uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
        xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)

        client = hl2ss.rx_decoded_rm_depth_longthrow(
            host, 
            hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
            hl2ss.ChunkSize.RM_DEPTH_LONGTHROW,
            hl2ss.StreamMode.MODE_1,
            hl2ss.PngFilterMode.Paeth
        )
        
        time.sleep(5)
        
        client.open()

        start_t = None

        while True:
            data = client.get_next_packet()
            current_t = to_timestamp(data.timestamp + utc_offset)
            
            if start_t is None:
                start_t = current_t
                
            print(current_t, int(time.time() * 1000))

            depth = hl2ss_3dcv.rm_depth_undistort(data.payload.depth, calibration_lt.undistort_map)
            depth = hl2ss_3dcv.rm_depth_normalize(depth, scale)
            lt_points = hl2ss_3dcv.rm_depth_to_points(xy1, depth)
            lt_to_world = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data.pose)
            
            source = lt_points.reshape(-1, 3)
            np.savez(f"{output_dir}/{current_t}_device_hololens.npz", xyz=source, pose=lt_to_world.T)
            
            cv2.imshow('Depth', data.payload.depth / np.max(data.payload.depth)) # Scaled for visibility
            
            if current_t - start_t > 20000:
                cv2.destroyAllWindows()
                break
            
            key = cv2.waitKey(1)
            if key in (27, ord("q")):
                cv2.destroyAllWindows()
                break

    finally:
        client.close()
        controller.send("global_lidar stop")
        rx.join()
    

if __name__ == "__main__":
    main()