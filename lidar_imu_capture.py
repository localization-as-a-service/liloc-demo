import argparse
import zmq
import time
import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
import multiprocessing as mp
import utils.pointcloud as pointcloud

from threading import Thread


class GPCRequester(mp.Process):
    def __init__(self, queue: mp.Queue, address: str = "localhost", port: int = 5556):
        super(GPCRequester, self).__init__()
        self.queue = queue
        self.url = f"tcp://{address}:{port}"
        
    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.connect(self.url)
        print("Connected to GPC server")
        
        while True:
            try:
                data, timestamp = self.queue.get()
                socket.send_string("send_gpc")
            except KeyboardInterrupt:
                break
            
        socket.close()


def extract_motion_data(timestamp, accel_data, gyro_data):
    return np.array([
        timestamp,
        accel_data.x, accel_data.y, accel_data.z,
        gyro_data.x, gyro_data.y, gyro_data.z
    ], dtype=np.float32)


def send_array(socket, array, sensor, timestamp, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(array.dtype),
        shape=array.shape,
        timestamp=timestamp,
        sensor=sensor
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(array, flags, copy=copy, track=track)

    
def imu_stream():
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://192.168.10.101:5555")
    
    imu_pipe = rs.pipeline()
    imu_config = rs.config()

    imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
    imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    
    imu_pipe.start(imu_config)
    
    start_t = time.time()
    previous_t = time.time() * 1000
    elapsed_t = 0
    
    while True:
        try:
            elapsed_t = time.time() - start_t
            
            if elapsed_t > 27: break
            
            motion_frames: rs.composite_frame = imu_pipe.poll_for_frames()

            if motion_frames:
                current_t = motion_frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
                
                if current_t - previous_t < 1: continue
                
                fps = int(1 / (current_t - previous_t + 1) * 1e3)
                previous_t = current_t
                
                accel_frame = motion_frames[0].as_motion_frame()
                gyro_frame = motion_frames[1].as_motion_frame()
                
                data = extract_motion_data(current_t, accel_frame.get_motion_data(), gyro_frame.get_motion_data())
                
                send_array(socket, data, 0, current_t)
                # socket.recv()
                
                print(f"Capturing IMU data @ {fps} Hz", end="\r")
                
        except KeyboardInterrupt:
            break
    
    send_array(socket, np.zeros(7), 2, 0)
    # socket.recv()
        
    imu_pipe.stop()
    
def camera_stream():
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://192.168.10.101:5555")
    
    lpc_queue = mp.Queue()
    gpc_requester = GPCRequester(lpc_queue)
    gpc_requester.start()
    
    camera_pipe = rs.pipeline()
    camera_config = rs.config()
    
    camera_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pc = rs.pointcloud()
    
    camera_pipe.start(camera_config)
    
    previous_t = time.time() * 1000
    global_t = time.time() * 1000
    start_t = time.time()
    elapsed_t = 0
    
    while True:
        try:
            elapsed_t = time.time() - start_t
            
            if elapsed_t > 25:
                break
            
            frames = camera_pipe.wait_for_frames()

            depth_frame = frames.get_depth_frame() 
            
            if not depth_frame:
                continue
            
            vertices = pc.calculate(depth_frame).get_vertices()
            vertices = np.asanyarray(vertices).view(np.float32).reshape(-1, 3)
            
            # valid_vertices = np.logical_and(np.logical_and(vertices[:, 0] > 0.25, vertices[:, 1] > 0.25), vertices[:, 2] > 0.25)
            # vertices = vertices[valid_vertices]
            # pcd = pointcloud.make_pcd(vertices)
            # pcd = pointcloud.downsample(pcd, 0.05)
            # vertices = np.asarray(pcd.points)
            
            indices = np.random.choice(np.arange(vertices.shape[0]), 10000, replace=False)
            vertices_sampled = vertices[indices]
            
            current_t = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
            
            fps = int(1 / (current_t - previous_t + 1) * 1e3)
            previous_t = current_t
            
            calibrating = elapsed_t < 5
            
            message = f"FPS: {fps:2d} | Elapsed Time: {elapsed_t:03.2f}s | " + ("Calibrating" if calibrating else "Recording")
            
            if not calibrating:
                send_array(socket, vertices_sampled, 1, current_t)
                # socket.recv()
                if current_t - global_t > 800:
                    print(f"Sending LPC for global registration after {current_t - global_t} ms")
                    lpc_queue.put((vertices, current_t))
                    global_t = current_t

            depth_frame = np.asanyarray(depth_frame.get_data())
            # normalize depth frame
            depth_frame = np.array(depth_frame / 65535 * 255, dtype=np.uint8)
            depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
            cv2.putText(depth_frame, message, (7, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 76, 231), 2, cv2.LINE_AA)
            cv2.namedWindow("Secondary View", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Secondary View", depth_frame)

            # To get key presses
            key = cv2.waitKey(1)

            if key in (27, ord("q")) or cv2.getWindowProperty("Secondary View", cv2.WND_PROP_AUTOSIZE) < 0:
                break
            
        except KeyboardInterrupt:
            break

    camera_pipe.stop()
    gpc_requester.terminate()
    gpc_requester.join()
    cv2.destroyAllWindows()
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Intel Lidar L515")
    parser.add_argument("--mode", default="cam", type=str)
    args = parser.parse_args()

    if args.mode == "cam":
        camera_stream()
    elif args.mode == "imu":
        imu_stream()
    elif args.mode == "all":
        Thread(target=imu_stream, args=()).start()
        Thread(target=camera_stream, args=()).start()
    else:
        print("Unsuppored mode.")