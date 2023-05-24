#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import open3d
import zmq
import numpy as np
import cv2


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md["dtype"])
    return A.reshape(md["shape"])


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# cv2.namedWindow("LiDAR Viewer", cv2.WINDOW_AUTOSIZE)
# cv2.resizeWindow("LiDAR Viewer", 640, 480)

# vis = open3d.visualization.Visualizer()
# vis.create_window()

# global_pcd = open3d.geometry.PointCloud()
# global_pcd.points = open3d.utility.Vector3dVector(np.random.rand(100, 3))

# vis.add_geometry(global_pcd)

while True:
    depth_frame = recv_array(socket)
    cv2.imshow("Depth Camera", depth_frame)
    print("Received depth frame: %s" % depth_frame.shape)
    socket.send(b"OK")
    
    # try:
    #     pcd = recv_array(socket)
    #     print(f"Received point cloud: {pcd.shape}")
        
        # global_pcd.points = open3d.utility.Vector3dVector(pcd)
        # # global_pcd.paint_uniform_color([0, 0.651, 0.929])
        # vis.update_geometry()
        # vis.poll_events()
        # vis.update_renderer()
        
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

    
    # except KeyboardInterrupt:
    #     break
    
socket.close()
# vis.destroy_window()


