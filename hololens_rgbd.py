import sys

sys.path.append('./hl2ss/Comp_Engine')

import hl2ss_imshow
import hl2ss
import hl2ss_3dcv
import hl2ss_mp

import zmq
import cv2
import time
import numpy as np
import multiprocessing as mp

from collections import deque
from threading import Thread
from zmq.utils.monitor import recv_monitor_message
from utils.depth_camera import DepthCamera, DepthCameraParams
from utils.local_registration import LocalRegistration


def to_timestamp(timestamp):
    return int((timestamp - 116444736000000000) / 10000)
        

def main():
    # HoloLens address
    host = "192.168.10.149"
    calibration_path = './hl2ss/calibration'
    
    num_devices = 3
    utc_offset = 133438974452878952 # client_rc.py to get offset
    
    # Video encoding profile for AHAT
    profile = hl2ss.VideoProfile.H265_MAIN
    # Buffer length in seconds
    buffer_length = 10
    # Front RGB camera parameters
    width = 640
    height = 360
    framerate = 30
    # Maximum depth in meters
    max_depth = 8.0
    # Encoded stream average bits per second for AHAT
    bitrate = hl2ss.get_video_codec_bitrate(width, height, framerate, hl2ss.get_video_codec_default_factor(profile))
    
    calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)

    uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)

    # client = hl2ss.rx_decoded_rm_depth_longthrow(
    #     host, 
    #     hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
    #     hl2ss.ChunkSize.RM_DEPTH_LONGTHROW,
    #     hl2ss.StreamMode.MODE_1,
    #     hl2ss.PngFilterMode.Paeth
    # )
    
    # Start PV and RM Depth Long Throw streams
    producer = hl2ss_mp.producer()
    producer.configure_pv(True, host, hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss.ChunkSize.PERSONAL_VIDEO, hl2ss.StreamMode.MODE_1, width, height, framerate, profile, bitrate, 'rgb24')
    producer.configure_rm_depth_longthrow(True, host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.ChunkSize.RM_DEPTH_LONGTHROW, hl2ss.StreamMode.MODE_1, hl2ss.PngFilterMode.Paeth)
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, framerate * buffer_length)
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS * buffer_length)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, ...)
    
    sink_pv.get_attach_response()
    sink_depth.get_attach_response()
    
    # Initialize PV intrinsics and extrinsics
    pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)
    
    start_t = None
    global_t = None

    while True:
        # Wait for RM Depth Long Throw frame
        sink_depth.acquire()
        
        # Get RM Depth Long Throw frame and nearest (in time) PV frame --------
        _, data_lt = sink_depth.get_most_recent_frame()
        if ((data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose))):
            continue

        _, data_pv = sink_pv.get_nearest(data_lt.timestamp)
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            continue
        

        current_t = to_timestamp(data_lt.timestamp + utc_offset)
        print(current_t, int(time.time() * 1000))
        
        if start_t is None:
            start_t = current_t
            
        if global_t is None:
            global_t = current_t
        

        # Preprocess frames ---------------------------------------------------
        color = data_pv.payload.image
        
        depth = hl2ss_3dcv.rm_depth_undistort(data_lt.payload.depth, calibration_lt.undistort_map)
        depth = hl2ss_3dcv.rm_depth_normalize(depth, scale)
        
        # Update PV intrinsics
        # PV intrinsics may change between frames due to autofocus
        pv_intrinsics = hl2ss.update_pv_intrinsics(pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point)
        color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)
        
        lt_points = hl2ss_3dcv.rm_depth_to_points(xy1, depth)
        lt_to_world = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
        
        # source = lt_points.reshape(-1, 3)
        
        # world_to_lt = hl2ss_3dcv.world_to_reference(data_lt.pose) @ hl2ss_3dcv.rignode_to_camera(calibration_lt.extrinsics)
        world_to_pv_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)
        world_points = hl2ss_3dcv.transform(lt_points, lt_to_world)
        pv_uv = hl2ss_3dcv.project(world_points, world_to_pv_image)
        color = cv2.remap(color, pv_uv[:, :, 0], pv_uv[:, :, 1], cv2.INTER_LINEAR)
        
        mask_uv = hl2ss_3dcv.slice_to_block((pv_uv[:, :, 0] < 0) | (pv_uv[:, :, 0] >= width) | (pv_uv[:, :, 1] < 0) | (pv_uv[:, :, 1] >= height))
        depth[mask_uv] = 0
        
        # lr.update(current_t, source, lt_to_world.T)
        
        # cv2.imshow('Depth', data.payload.depth / np.max(data.payload.depth)) # Scaled for visibility
        
        # if current_t - global_t > 1000:
        #     print(f"Triggering global registration after {current_t - global_t} ms")
            
        #     target = rx.get_global_pcd(current_t)
                
        #     send_data(socket, current_t, source=source, target=target)
            
        #     global_t = current_t
        
        # if current_t - start_t > 20000:
        #     cv2.destroyAllWindows()
        #     break
        
        
        cv2.imshow("Color", color)
        cv2.imshow("Depth", depth / max_depth)
        
        key = cv2.waitKey(1)
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            break
        

    # Stop PV and RM Depth Long Throw streams ---------------------------------
    sink_pv.detach()
    sink_depth.detach()
    producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
    
    # Stop PV subsystem -------------------------------------------------------
    hl2ss.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)


if __name__ == "__main__":
    main()