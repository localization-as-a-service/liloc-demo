import zmq
import sys
import time
import zmq
import numpy as np
import requests

from flask import Flask, request, jsonify
from pynput import keyboard
from threading import Thread

from utils.depth_camera import DepthCamera, DepthCameraParams
from utils.buffers import TimestampArrayBuffer
from zmq.utils.monitor import recv_monitor_message


class GlobalLiDARController:
    def __init__(self, num_devices: int, port: int=5556):
        self.num_devices = num_devices
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        
        connections = 0
        events_socket = self.socket.get_monitor_socket(events=zmq.EVENT_HANDSHAKE_SUCCEEDED)
        
        while connections < self.num_devices:
            # this will block until a handshake was successful
            recv_monitor_message(events_socket)  
            connections += 1
            print(f"Connection established to device: {connections}")
        
    def send(self, message: str):
        self.socket.send_string(message)
            
    def close(self):
        self.socket.close()


class DepthImageRx(Thread):
    def __init__(self, num_devices: int, staitc_env_file: str, buffer_size: int=600):
        super(DepthImageRx, self).__init__()
        self.num_devices = num_devices
        self.buffers = [TimestampArrayBuffer(max_buffer_size=buffer_size) for _ in range(self.num_devices)]
        self.cameras = []
        self.transformations = []
        self.running = True

        if staitc_env_file is not None:
            self.static_pts = np.load(staitc_env_file)
        
        for i in range(self.num_devices):
            camera_params = DepthCameraParams(f"metadata/device-{i}.json")
            depth_camera = DepthCamera(camera_params)
            transformation = np.loadtxt(f"metadata/device-{i}.txt")

            self.cameras.append(depth_camera)
            self.transformations.append(transformation)
            
        print("Starting Depth Image Rx.")
        
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
        
        while self.running:
            try:
                depth_image, timestamp, sensor = self.recv_array(socket)
                self.buffers[sensor].add_data(timestamp, depth_image)
                
                time.sleep(0.001)
            except KeyboardInterrupt:
                break
            
        socket.close()

    def get_latest_depth_image(self, sensor):
        return self.buffers[sensor].get_latest()
    
    def get_global_pcd(self, timestamp):
        targets = []
        for i in range(self.num_devices):
            _, frame = self.buffers[i].find_nearest(timestamp)
            vertices = self.cameras[i].depth_image_to_point_cloud(frame)
            vertices = np.concatenate((vertices, np.ones((vertices.shape[0], 1))), axis=1)
            vertices = np.dot(self.transformations[i], vertices.T).T
            vertices = vertices[:, :3]
            
            targets.append(vertices)

        targets.append(self.static_pts)
            
        return np.vstack(targets).astype(np.float32)
    
    def close(self):
        self.running = False

    def is_running(self):
        return self.running
    

num_devices = 3

controller = GlobalLiDARController(num_devices)
receiver = DepthImageRx(num_devices, staitc_env_file="../calibration/env_static.npy")

app = Flask(__name__)


@app.route("/health-check")
def health_check():
    return "OK"

@app.route("/start")
def start():
    controller.send("global_lidar start")
    return "OK"

@app.route("/stop")
def stop():
    controller.send("global_lidar stop")
    return "OK"

@app.route("/latest-depth-image")
def latest_depth_image():
    response = {}
    for i in range(num_devices):
        data = receiver.get_latest_depth_image(i)
        response[f"device-{i}"] = {
            "timestamp": data[0],
            "image": data[1].tolist()
        }
    
    return jsonify(response)

@app.route("/global-pcd")
def global_pcd():
    timestamp = int(request.args.get("timestamp"))
    pcd = receiver.get_global_pcd(timestamp)
    return jsonify(pcd.tolist())

@app.route("/shutdown")
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


def on_press(key):
    if key == keyboard.Key.esc:
        print("ESC pressed, exiting...")
        controller.send("global_lidar stop")

        # requests.get("http://localhost:5000/shutdown")
        
        if receiver.is_running():
            receiver.close()
            receiver.join()

        sys.exit(0)

if __name__ == "__main__":

    # listener = keyboard.Listener(on_press=on_press)
    # listener.start()
    
    receiver.start()

    app.run(host="0.0.0.0", port=5000)

    
    
    