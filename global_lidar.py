import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import time
import zmq
from threading import Lock


address = "10.4.20.7"
port = 5002
device = 2
lock = Lock()

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect(f"tcp://{address}:{port}")

def send_array(data, timestamp, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(data.dtype),
        shape=data.shape,
        timestamp=timestamp,
        device=device
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    socket.send(data, flags, copy=copy, track=track)


def frame_handler(frames):
    frame_set = frames.as_frameset()
    depth_frame = frame_set.get_depth_frame() 
    current_t = frame_set.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)

    if not depth_frame:
        return

    lock.acquire()

    depth_image = np.asanyarray(depth_frame.get_data())
    #depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    send_array(depth_image, current_t)

    print(f"Sending frame for {current_t}", end="\r")

    lock.release()

ps_socket = context.socket(zmq.SUB)
ps_socket.setsockopt_string(zmq.SUBSCRIBE, "global_lidar")
ps_socket.connect(f"tcp://{address}:5001")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

started = False

while True:
    try:
        topic, message = ps_socket.recv_string().split(" ", 1)
        print(topic, message)
        if message == "start":
            if started:
                continue
            else:
                print("Starting the camera @", time.time())
                pipeline.start(config, frame_handler)
                started = True
        elif message == "stop":
            if started:
                print("Stopping the camera @", time.time())
                pipeline.stop()
                started = False
        else:
            print("unsupported command.")
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        break

ps_socket.close()
socket.close()
context.term()
