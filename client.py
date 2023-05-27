import os
import zmq
import time
import open3d
import numpy as np

from threading import Thread


def send_array(socket, array, idx, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(array.dtype),
        shape=array.shape,
        idx=idx
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(array, flags, copy=copy, track=track)


def read_point_cloud(filename):
    pcd = open3d.io.read_point_cloud(filename)
    return np.asarray(pcd.points)


def main(url):
    ctx = zmq.Context()
    s = ctx.socket(zmq.PUSH)
    sequence_ts = ['1680509270952', '1680509271019', '1680509271085', '1680509271118']

    s.connect(url)

    for i in sequence_ts:
        print("Sending LPC @ {}".format(i))
        send_array(s, read_point_cloud(f"temp/global_reg/{i}.secondary.pcd"), 0)
        time.sleep(0.05)
        print("Sending GPC @ {}".format(i))
        send_array(s, read_point_cloud(f"temp/global_reg/{i}.global.pcd"), 1)
        time.sleep(0.8)
        
    print("Done")
    s.close()
    
if __name__ == '__main__':
    main('tcp://localhost:5557')