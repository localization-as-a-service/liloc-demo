import open3d
import numpy as np
import pandas as pd
import os
import glob
import tqdm
import copy
import zmq

import utils.pointcloud as pointcloud
import utils.registration as registration
import utils.grid_search_rs_unopt as grid_search


def recv_array(socket, flags=0, copy=True, track=False):
    data = socket.recv_json(flags=flags)
    timestamp = data["timestamp"]
    pcd = pointcloud.make_pcd(np.array(data["points"]))
    transformation = np.array(data["transformation"])
    token = data["token"]
    return timestamp, pcd, transformation, token


def main():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5559")
    
    while True:
        try:
            timestamp, pcd, transformation, token = recv_array(socket)
            print(f"Timestamp: {timestamp} | Token: {token}")
        except KeyboardInterrupt:
            break
        
    socket.close()

if __name__ == "__main__":
    main()