import platform
import multiprocessing as mp
import numpy as np
import argparse
import zmq

from time import time, sleep

if platform.uname().machine == 'aarch64':
    import pyrealsense2.pyrealsense2 as rs
else:
    import pyrealsense2 as rs


class GlobalLidar(mp.Process):
    
    def __init__(self, queue: mp.Queue, device: int = 0, address: str = 'localhost', port: int = 5554):
        super(GlobalLidar, self).__init__()
        self.device = device
        self.url = f"tcp://{address}:{port}"
        self.queue = queue
        
    def send_array(self, data, timestamp, flags=0, copy=True, track=False):
        """send a numpy array with metadata"""
        md = dict(
            dtype=str(data.dtype),
            shape=data.shape,
            timestamp=timestamp,
            device=self.device
        )
        self.socket.send_json(md, flags | zmq.SNDMORE)
        self.socket.send(data, flags, copy=copy, track=track)
        
    def run(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(self.url)
        print("Connected to LiLOC server @ ", self.url)
        
        try:
            while True:
                try:
                    depth_frame, timestamp = None, None
                    while not self.queue.empty():
                        depth_frame, timestamp = self.queue.get()
                        
                    if depth_frame is None or timestamp is None:
                        continue
                    
                    # depth_frame, timestamp = self.queue.get()
                    self.send_array(depth_frame, timestamp)
                    
                    sleep(0.005)
                except KeyboardInterrupt:
                    break
        finally:
            self.socket.close()
                        

def main(args, queue, process):
    if args.resolution == 640:
        width, height = 640, 480
    elif args.resolution == 1280:
        width, height = 1280, 768
    else:
        print("Invalid resolution! The resolution must be either 640 or 1280")

    pipeline = rs.pipeline()
    
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    
    pipeline.start(config)
    
    print(f"Starting Recording @ {int(time())}")

    try:
        previous_t = time() * 1000
        start_t = previous_t

        while True:
            try:
                frames = pipeline.wait_for_frames()

                if not frames:
                    continue

                depth_frame = np.asanyarray(frames.get_depth_frame().get_data())
                
                current_t = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
                
                if current_t - previous_t < 80: continue
                
                if current_t - start_t > 60000: break
                
                fps = int(1 / (current_t - previous_t + 1) * 1e3)
                previous_t = current_t

                queue.put((depth_frame, current_t))
                
                print(f"Capturing images @ {fps:02d} fps @ {current_t}", end="\r")

                sleep(0.005)
            except KeyboardInterrupt:
                break
    finally:
        pipeline.stop()
        print(f"Stopped Recording @ {int(time())}")
        process.terminate()
        process.join()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lidar Capture for Intel RealSense L515')
    parser.add_argument('-r', '--resolution', type=int, default=640, help='Resolution of the image')
    parser.add_argument('-d', '--device', type=int, default=0, help='Device ID')
    parser.add_argument('-a', '--address', type=str, default='localhost', help='IP Address of the server')
    parser.add_argument('-p', '--port', type=int, default=5554, help='Port of the server')

    args = parser.parse_args()
    
    queue = mp.Queue()
    gl = GlobalLidar(queue, args.device, args.address, args.port)
    gl.start()
    
    main(args, queue)