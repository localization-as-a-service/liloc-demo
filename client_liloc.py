from pynput import keyboard
from utils.local_registration import LocalRegistration
import cv2
import zmq
import time
import open3d
import numpy as np

import sys

sys.path.append('./hl2ss/Comp_Engine')

import hl2ss_imshow
import hl2ss
import hl2ss_3dcv



def to_timestamp(timestamp):
    return int((timestamp - 116444736000000000) / 10000)
        
def main():
    # HoloLens address
    host = "192.168.10.149"
    calibration_path = './hl2ss/calibration'
    utc_offset = 133438974452878952 # client_rc.py to get offset
    
    lr = LocalRegistration()

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
    
    
    client.open()

    start_t = None
    global_t = None

    while True:
        data = client.get_next_packet()
        current_t = to_timestamp(data.timestamp + utc_offset)
        
        if start_t is None:
            start_t = current_t
            
        if global_t is None:
            global_t = current_t
        
        print(current_t, int(time.time() * 1000))

        depth = hl2ss_3dcv.rm_depth_undistort(data.payload.depth, calibration_lt.undistort_map)
        depth = hl2ss_3dcv.rm_depth_normalize(depth, scale)
        lt_points = hl2ss_3dcv.rm_depth_to_points(xy1, depth)
        lt_to_world = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data.pose)
        # world_points = hl2ss_3dcv.transform(lt_points, lt_to_world)
        
        vertices = lt_points.reshape(-1, 3)
        lr.update(current_t, vertices, lt_to_world.T)
        
        cv2.imshow('Depth', data.payload.depth / np.max(data.payload.depth)) # Scaled for visibility
        
        print(current_t - global_t)
        
        if current_t - global_t > 800:
            print(f"Triggering global registration after {current_t - global_t} ms")
            global_t = current_t
        
        if current_t - start_t > 30000:
            cv2.destroyAllWindows()
            break
        
        key = cv2.waitKey(1)
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            break

    client.close()
    
    x = lr.get_trajectory()
    open3d.visualization.draw_geometries([x])
    

if __name__ == "__main__":
    main()