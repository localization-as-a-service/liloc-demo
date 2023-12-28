#------------------------------------------------------------------------------
# This script demonstrates how to create aligned RGBD images, which can be used
# with Open3D, from the depth and front RGB cameras of the HoloLens.
# Press space to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import multiprocessing as mp
import numpy as np
import open3d
import cv2
import time
from hl2ss import hl2ss_imshow
from hl2ss import hl2ss
from hl2ss import hl2ss_lnm
from hl2ss import hl2ss_mp
from hl2ss import hl2ss_3dcv
from threading import Thread

from utils.transform import inv_transform
from utils.interfaces import LiLOCServiceInterface


class Visualizer(Thread):
    def __init__(self):
        super(Visualizer, self).__init__()
        self.vis = open3d.visualization.Visualizer()
        self.global_pcd = None
        self.local_pcd = None
        self.line_set = None
        self.running = True
    
    def set_global_pcd(self, xyz):
        if self.global_pcd is not None:
            self.global_pcd.points = open3d.utility.Vector3dVector(xyz)
            self.global_pcd.paint_uniform_color([0, 0.651, 0.929])
        else:
            self.global_pcd = open3d.geometry.PointCloud()
            self.global_pcd.points = open3d.utility.Vector3dVector(xyz)
            self.global_pcd.paint_uniform_color([0, 0.651, 0.929])
            self.vis.add_geometry(self.global_pcd)

    def set_local_pcd(self, xyz):
        if self.local_pcd is not None:
            self.local_pcd.points = open3d.utility.Vector3dVector(xyz)
            self.local_pcd.paint_uniform_color([1, 0.706, 0])
        else:
            self.local_pcd = open3d.geometry.PointCloud()
            self.local_pcd.points = open3d.utility.Vector3dVector(xyz)
            self.local_pcd.paint_uniform_color([1, 0.706, 0])
            self.vis.add_geometry(self.local_pcd)

    def draw_trajectory(self, global_t):
        trajectory = np.array([global_t[i][:3, 3] for i in range(len(global_t)) if np.sum(global_t[i]) != 4])
        lines = [[i, i + 1] for i in range(len(trajectory) - 1)]
        colors = [[1, 0, 0] for _ in range(len(lines))]

        if self.line_set is not None:
            self.line_set.points = open3d.utility.Vector3dVector(trajectory)
            self.line_set.lines = open3d.utility.Vector2iVector(lines)
            self.line_set.colors = open3d.utility.Vector3dVector(colors)
        else:
            self.line_set = open3d.geometry.LineSet()

            self.line_set.points = open3d.utility.Vector3dVector(trajectory)
            self.line_set.lines = open3d.utility.Vector2iVector(lines)
            self.line_set.colors = open3d.utility.Vector3dVector(colors)

            self.vis.add_geometry(self.line_set)

    def run(self):
        self.vis.create_window()

        while self.running:
            self.vis.update_geometry()
            self.vis.poll_events()
            self.vis.update_renderer()

            time.sleep(0.005)

        self.vis.destroy_window()

    def stop(self):
        self.running = False


def get_utc_offset(host):
    try:
        client = hl2ss_lnm.ipc_rc(host, hl2ss.IPCPort.REMOTE_CONFIGURATION)
        client.open()
        return client.get_utc_offset(32)
    finally:
        client.close()


def update_visualizer_global(liloc_server, vis, current_t, source):
    target = liloc_server.update_global(current_t, source)
    vis.set_global_pcd(target)


def update_visualizer_local(vis, source, transformation_matrix):
    if np.sum(transformation_matrix) == 4:
        return

    source = np.concatenate((source, np.ones((source.shape[0], 1))), axis=1)
    source = np.dot(transformation_matrix, source.T).T

    vis.set_local_pcd(source[:, :3])


def to_timestamp(timestamp):
    return int((timestamp - 116444736000000000) / 10000)


def project_points_to_image(points, world_to_pv_image, color):
    xyz = hl2ss_3dcv.transform(points, world_to_pv_image)
    z = xyz[:, 2]

    u = np.array(xyz[:, 0] / z)
    v = np.array(xyz[:, 1] / z)

    depth_image = np.zeros((color.shape[0], color.shape[1]))

    indices_out_of_bounds = np.logical_or(u < 0, u >= depth_image.shape[1])
    indices_out_of_bounds = np.logical_or(indices_out_of_bounds, v < 0)
    indices_out_of_bounds = np.logical_or(indices_out_of_bounds, v >= depth_image.shape[0])

    u = np.delete(u, indices_out_of_bounds)
    v = np.delete(v, indices_out_of_bounds)
    z = np.delete(z, indices_out_of_bounds)

    u = u.astype(int)
    v = v.astype(int)

    depth_image[v, u] = z

    return depth_image


def main():
    # Settings --------------------------------------------------------------------

    # HoloLens address
    host = '192.168.10.149'

    # Calibration path (must exist but can be empty)
    calibration_path = './calib'
    session_id = time.strftime("%Y%m%d%H%M%S")
    
    utc_offset = get_utc_offset(host)
    liloc_server = LiLOCServiceInterface("http://localhost:5000", session_id)

    # Front RGB camera parameters
    pv_width = 640
    pv_height = 360
    pv_framerate = 30

    # Buffer length in seconds
    buffer_length = 10

    # Maximum depth in meters
    max_depth = 5.0

    # Keyboard events ---------------------------------------------------------
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.space
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        if not liloc_server.start():
            print("Failed to start global lidar server")
            return

        time.sleep(5)

        device_local_t = [np.identity(4)]
        device_global_t = []
        
        vis = Visualizer()
        vis.start()

        # Start PV Subsystem ------------------------------------------------------
        hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

        # Get RM Depth Long Throw calibration -------------------------------------
        # Calibration data will be downloaded if it's not in the calibration folder
        calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)

        uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
        xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)

        # Start PV and RM Depth Long Throw streams --------------------------------
        producer = hl2ss_mp.producer()
        producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate, decoded_format='rgb24'))
        producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
        producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_framerate * buffer_length)
        producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS * buffer_length)
        producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
        producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

        consumer = hl2ss_mp.consumer()
        manager = mp.Manager()
        sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
        sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, ...)

        sink_pv.get_attach_response()
        sink_depth.get_attach_response()

        # Initialize PV intrinsics and extrinsics ---------------------------------
        pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
        pv_extrinsics = np.eye(4, 4, dtype=np.float32)

        start_t = None
        global_t = None
    
        # Main Loop ---------------------------------------------------------------
        while (enable):
            # Wait for RM Depth Long Throw frame ----------------------------------
            sink_depth.acquire()

            # Get RM Depth Long Throw frame and nearest (in time) PV frame --------
            _, data_lt = sink_depth.get_most_recent_frame()
            if ((data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose))):
                continue

            _, data_pv = sink_pv.get_nearest(data_lt.timestamp)
            if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
                continue

            current_t = to_timestamp(data_lt.timestamp + utc_offset)
            
            if start_t is None:
                start_t = current_t
                
            if global_t is None:
                global_t = current_t

            # Preprocess frames ---------------------------------------------------
            depth = hl2ss_3dcv.rm_depth_undistort(data_lt.payload.depth, calibration_lt.undistort_map)
            depth = hl2ss_3dcv.rm_depth_normalize(depth, scale)
            color = data_pv.payload.image

            # Update PV intrinsics ------------------------------------------------
            # PV intrinsics may change between frames due to autofocus
            pv_intrinsics = hl2ss.update_pv_intrinsics(pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point)
            color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)
            
            # Generate aligned RGBD image -----------------------------------------
            lt_points         = hl2ss_3dcv.rm_depth_to_points(xy1, depth)
            lt_to_world       = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
            # world_to_lt       = hl2ss_3dcv.world_to_reference(data_lt.pose) @ hl2ss_3dcv.rignode_to_camera(calibration_lt.extrinsics)
            world_to_pv_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)
            world_points      = hl2ss_3dcv.transform(lt_points, lt_to_world)
            
            projected_depth = project_points_to_image(world_points.reshape(-1, 3), world_to_pv_image, color)
            
            source = lt_points.reshape(-1, 3)
            
            device_global_t.append(lt_to_world.T)
            
            if len(device_global_t) > 1:
                t = np.dot(inv_transform(device_global_t[-2]), device_global_t[-1])
                device_local_t.append(t)

            source_t = liloc_server.update_local(current_t, device_local_t[-1])
            update_visualizer_local(vis, source, source_t)

            # Display RGBD --------------------------------------------------------
            image = np.hstack((color, cv2.applyColorMap((projected_depth / max_depth * 255).astype(np.uint8) , cv2.COLORMAP_JET)))
            cv2.imshow('RGBD', image)
            
            if current_t - global_t > 1200:
                print(f"Triggering global registration after {current_t - global_t} ms")

                Thread(target=update_visualizer_global, args=(liloc_server, vis, current_t, source)).start()

                global_t = current_t
            
            if current_t - start_t > 600000:
                cv2.destroyAllWindows()
                break
            
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
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

        # Stop keyboard events ----------------------------------------------------
        listener.join()
    
    finally:
        liloc_server.stop()
        vis.stop()


#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()