import zmq
import open3d
import time
import numpy as np
import utils.grid_search_rs_unopt as grid_search
import utils.registration as registration
import utils.pointcloud as pointcloud

import multiprocessing as mp


class GlobalRegistration(mp.Process):
    
    def __init__(self, stop_event: mp.Value):
        super(GlobalRegistration, self).__init__()
        self.stop_event = stop_event
        
    def _receive_array(self, socket, flags=0, copy=True, track=False):
        md = socket.recv_json(flags=flags)
        return md['timestamps'], np.array(md['source']), np.array(md['source_feat']), np.array(md['target']), np.array(md['target_feat'])


    def _send_data(self, socket, timestamp, pcd, transformation, token):
        data = {
            "timestamp": timestamp,
            "vertices": np.asarray(pcd.points).tolist(),
            "transformation": transformation.tolist(),
            "token": token
        }
        socket.send_json(data, flags=0)

        
    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.connect("tcp://localhost:5558")
        
        gv_socket = context.socket(zmq.PUSH)
        gv_socket.connect("tcp://localhost:5559")
        
        while True:
            try:
                timestamps, source, source_feat, target, target_feat = self._receive_array(socket)
                print(f"{timestamps[0]} | Source: {source.shape} | Source Feat: {source_feat.shape} | Target: {target.shape} | Target Feat: {target_feat.shape}")

                source, target, result = grid_search.global_registration(source, source_feat, target, target_feat, cell_size=2, n_random=0.5, refine_enabled=True)
                registration.describe(source, target, result)
                
                self._send_data(gv_socket, timestamps[0], target, np.asarray(result.transformation), 1)
                
                if self.stop_event.value > 0:
                    print("Stopping global registration")
                    break
                
            except KeyboardInterrupt:
                break
            
        socket.close()
        gv_socket.close()


# def receive_array(socket, flags=0, copy=True, track=False):
#     md = socket.recv_json(flags=flags)
#     return md['timestamps'], np.array(md['source']), np.array(md['source_feat']), np.array(md['target']), np.array(md['target_feat'])


# def send_data(socket, timestamp, pcd, transformation, token):
#     data = {
#         "timestamp": timestamp,
#         "vertices": np.asarray(pcd.points).tolist(),
#         "transformation": transformation.tolist(),
#         "token": token
#     }
#     socket.send_json(data, flags=0)


# def main():
#     context = zmq.Context()
#     socket = context.socket(zmq.PAIR)
#     socket.connect("tcp://localhost:5558")
    
#     gv_socket = context.socket(zmq.PUSH)
#     gv_socket.connect("tcp://localhost:5559")
    
#     # vis = open3d.visualization.Visualizer()
#     # vis.create_window()

#     # global_pcd = open3d.geometry.PointCloud()
#     # global_pcd.points = open3d.utility.Vector3dVector(np.random.random((100, 3)))
#     # global_pcd.paint_uniform_color([1, 0.706, 0])
    
#     # local_pcd = open3d.geometry.PointCloud()
#     # local_pcd.points = open3d.utility.Vector3dVector(np.random.random((100, 3)))
#     # local_pcd.paint_uniform_color([0, 0.651, 0.929])
    
#     # vis.add_geometry(global_pcd)
#     # vis.add_geometry(local_pcd)
    
    
    
#     while True:
#         try:
#             timestamps, source, source_feat, target, target_feat = receive_array(socket)
#             print(f"{timestamps[0]} | Source: {source.shape} | Source Feat: {source_feat.shape} | Target: {target.shape} | Target Feat: {target_feat.shape}")
#             # np.savez_compressed(f"temp/trajectory/global_{timestamps[0]}.npz", source=source, source_feat=source_feat, target=target, target_feat=target_feat)            
            
#             # open3d.io.write_point_cloud(f"temp/source_{time.time_ns()}.pcd", pointcloud.make_pcd(source))
#             # open3d.io.write_point_cloud(f"temp/target_{time.time_ns()}.pcd", pointcloud.make_pcd(target))
#             source, target, result = grid_search.global_registration(source, source_feat, target, target_feat, cell_size=2, n_random=0.5, refine_enabled=True)
#             print(result)
            
#             send_data(gv_socket, timestamps[0], target, np.asarray(result.transformation), 1)
            
#             # registration.view(source, target, result.transformation)
            
#             # global_pcd.points = target.points
#             # global_pcd.paint_uniform_color([0, 0.651, 0.929])
            
#             # source.transform(result.transformation)
#             # local_pcd.points = source.points
#             # local_pcd.paint_uniform_color([1, 0.706, 0])
            
#             # vis.update_geometry()
#             # vis.poll_events()
#             # vis.update_renderer()
#             # time.sleep(0.005)
#         except KeyboardInterrupt:
#             break
        
#     socket.close()
#     gv_socket.close()

if __name__ == "__main__":
    # main()
    stop_event = mp.Value('i', 0)
    global_registration = GlobalRegistration(stop_event)
    global_registration.start()