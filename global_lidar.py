import pyrealsense2 as rs
import multiprocessing as mp
import numpy as np
# import cv2
import os
import argparse
import zmq

from time import time, sleep


def send_array(socket, data, timestamp, flags=0, copy=True, track=False):
        """send a numpy array with metadata"""
        md = dict(
            dtype=str(data.dtype),
            shape=data.shape,
            timestamp=timestamp
        )
        socket.send_json(md, flags | zmq.SNDMORE)
        return socket.send(data, flags, copy=copy, track=track)
    

def main(args):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{args.address}:{args.port}")
    
    print("Connected to LiLOC server...")
    
    if args.resolution == 640:
        width, height = 640, 480
    elif args.resolution == 1280:
        width, height = 1280, 768
    else:
        print("Invalid resolution! The resolution must be either 640 or 1280")

    pipeline = rs.pipeline()
    
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    
    # pcd = rs.pointcloud()
    
    pipeline.start(config)
    
    print(f"Starting Recording @ {int(time())}")

    try:
        previous_t = time() * 1000

        while True:
            try:
                frames = pipeline.wait_for_frames()

                if not frames:
                    continue

                depth_frame = frames.get_depth_frame() 
                
                depth_frame = np.asanyarray(depth_frame.get_data())
                
                # points = pcd.calculate(depth_frame).get_vertices()
                # vertices = np.asanyarray(points).view(np.float32).reshape(-1, 3)  # xyz
                # print(vertices.shape)

                current_t = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
                
                # if current_t - previous_t < 50: continue
                
                fps = int(1 / (current_t - previous_t + 1) * 1e3)
                previous_t = current_t

                send_array(socket, depth_frame, current_t)
                message = socket.recv()
                print("Received: ", message)

                # putting the FPS count on the frame
                # cv2.putText(color_image, f"{fps:2d} FPS", (7, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
                # cv2.namedWindow("Aligned RGB & Depth", cv2.WINDOW_AUTOSIZE)
                # cv2.imshow("Depth Camera", depth_image)

                # key = cv2.waitKey(1)

                # if key & 0xFF == ord('q') or key == 27:
                #     cv2.destroyAllWindows()

                print(f"Capturing images @ {fps:02d} fps @ {current_t}", end="\r")

                sleep(0.005)
            except KeyboardInterrupt:
                break
    finally:
        socket.close()
        pipeline.stop()
        
    
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lidar Capture for Intel RealSense L515')
    parser.add_argument('-r', '--resolution', type=int, default=640, help='Resolution of the image')
    parser.add_argument('-d', '--device', type=int, default=0, help='Device ID')
    parser.add_argument('-a', '--address', type=str, default='localhost', help='IP Address of the server')
    parser.add_argument('-p', '--port', type=int, default=5555, help='Port of the server')

    args = parser.parse_args()
    
    main(args)
    