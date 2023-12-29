import sys
import open3d
import time
import numpy as np
import cv2
import copy

from hl2ss import hl2ss_imshow, hl2ss, hl2ss_lnm, hl2ss_3dcv
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


def global_registration(glb_lidar, glb_reg_server, grv, current_t, source):
    target = glb_lidar.get_global_pcd(current_t)

    if not grv.found_correct_global:
        print(f"Performing global registration at {current_t}")
        global_t = glb_reg_server.global_registration(source, target)
        grv.update_global(current_t, global_t)


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
        

def main():
    # HoloLens address
    host = "192.168.10.149"
    calibration_path = './calib'
    session_id = time.strftime("%Y%m%d%H%M%S")
    
    utc_offset = get_utc_offset(host)
    liloc_server = LiLOCServiceInterface("http://localhost:5000", session_id)
    
    try:
        if not liloc_server.start():
            print("Failed to start global lidar server")
            return

        time.sleep(5)

        device_local_t = [np.identity(4)]
        device_global_t = []
        
        vis = Visualizer()
        vis.start()

        calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)

        uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
        xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)

        client = hl2ss.rx_decoded_rm_depth_longthrow(
            host=host, 
            port=hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
            chunk=hl2ss.ChunkSize.RM_DEPTH_LONGTHROW,
            mode=hl2ss.StreamMode.MODE_1,
            png_filter=hl2ss.PNGFilterMode.PAETH,
            divisor=1
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
            
            source = lt_points.reshape(-1, 3)
            
            device_global_t.append(lt_to_world.T)
            
            if len(device_global_t) > 1:
                t = np.dot(inv_transform(device_global_t[-2]), device_global_t[-1])
                device_local_t.append(t)

            source_t = liloc_server.update_local(current_t, device_local_t[-1])
            update_visualizer_local(vis, source, source_t)

            cv2.imshow('Depth', data.payload.depth / np.max(data.payload.depth)) # Scaled for visibility
            
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

    finally:
        client.close()
        liloc_server.stop()
        vis.stop()
    

if __name__ == "__main__":
    main()