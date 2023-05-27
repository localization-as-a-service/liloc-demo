import zmq
import time
import numpy as np
import open3d as o3d

def make_pcd(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd = o3d.geometry.voxel_down_sample(pcd, voxel_size)
    return pcd


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    array = np.frombuffer(buf, dtype=md["dtype"])
    return array.reshape(md["shape"]), md["idx"]


def main():
    voxel_size = 0.05
    # binding port for a socket
    ctx = zmq.Context()
    s = ctx.socket(zmq.PULL)
    s.bind('tcp://*:5557')
    
    while True:
        try:
            vertices, idx = recv_array(s)
            print("Received array: ", vertices.shape, vertices.dtype)
            pcd = make_pcd(vertices.copy(), voxel_size)
            o3d.visualization.draw_geometries([pcd])
        except KeyboardInterrupt:
            break
    print("Server Done")
    s.close()
    
if __name__ == '__main__':
    main()