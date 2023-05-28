import zmq
import open3d
import time
import numpy as np
import utils.grid_search_rs_unopt as grid_search
import utils.registration as registration
import utils.pointcloud as pointcloud


# def receive_array(socket, flags=0, copy=True, track=False):
#     md = socket.recv_json(flags=flags)
    
#     source_buffer = socket.recv(flags=flags, copy=copy, track=track)
#     source = np.frombuffer(memoryview(source_buffer), dtype=md['dtype']).reshape(md['source'])
    
#     source_feat_buffer = socket.recv(flags=flags, copy=copy, track=track)
#     source_feat = np.frombuffer(memoryview(source_feat_buffer), dtype=md['dtype']).reshape(md['source_feat'])
    
#     target_buffer = socket.recv(flags=flags, copy=copy, track=track)
#     target = np.frombuffer(memoryview(target_buffer), dtype=md['dtype']).reshape(md['target'])
    
#     target_feat_buffer = socket.recv(flags=flags, copy=copy, track=track)
#     target_feat = np.frombuffer(memoryview(target_feat_buffer), dtype=md['dtype']).reshape(md['target_feat'])
    
#     return source, source_feat, target, target_feat

def receive_array(socket, flags=0, copy=True, track=False):
    md = socket.recv_json(flags=flags)
    return md['timestamps'], np.array(md['source']), np.array(md['source_feat']), np.array(md['target']), np.array(md['target_feat'])


def send_data(socket, timestamp, pcd, transformation, token):
    data = {
        "timestamp": timestamp,
        "vertices": np.asarray(pcd.points).tolist(),
        "transformation": transformation.tolist(),
        "token": token
    }
    socket.send_json(data)


def main():
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://localhost:5558")
    
    gv_socket = context.socket(zmq.PUSH)
    gv_socket.connect("tcp://localhost:5559")
    
    # vis = open3d.visualization.Visualizer()
    # vis.create_window()

    # global_pcd = open3d.geometry.PointCloud()
    # global_pcd.points = open3d.utility.Vector3dVector(np.random.random((100, 3)))
    # global_pcd.paint_uniform_color([1, 0.706, 0])
    
    # local_pcd = open3d.geometry.PointCloud()
    # local_pcd.points = open3d.utility.Vector3dVector(np.random.random((100, 3)))
    # local_pcd.paint_uniform_color([0, 0.651, 0.929])
    
    # vis.add_geometry(global_pcd)
    # vis.add_geometry(local_pcd)
    
    
    
    while True:
        try:
            timestamps, source, source_feat, target, target_feat = receive_array(socket)
            print(f"{timestamps[0]} | Source: {source.shape} | Source Feat: {source_feat.shape} | Target: {target.shape} | Target Feat: {target_feat.shape}")
            # np.savez_compressed(f"temp/trajectory/global_{timestamps[0]}.npz", source=source, source_feat=source_feat, target=target, target_feat=target_feat)            
            
            # open3d.io.write_point_cloud(f"temp/source_{time.time_ns()}.pcd", pointcloud.make_pcd(source))
            # open3d.io.write_point_cloud(f"temp/target_{time.time_ns()}.pcd", pointcloud.make_pcd(target))
            source, target, result = grid_search.global_registration(source, source_feat, target, target_feat, cell_size=2, n_random=0.5, refine_enabled=True)
            print(result)
            
            send_data(gv_socket, timestamps[0], source, np.asarray(result.transformation), 1)
            
            # registration.view(source, target, result.transformation)
            
            # global_pcd.points = target.points
            # global_pcd.paint_uniform_color([0, 0.651, 0.929])
            
            # source.transform(result.transformation)
            # local_pcd.points = source.points
            # local_pcd.paint_uniform_color([1, 0.706, 0])
            
            # vis.update_geometry()
            # vis.poll_events()
            # vis.update_renderer()
            # time.sleep(0.005)
        except KeyboardInterrupt:
            break
        
    socket.close()
    gv_socket.close()

if __name__ == "__main__":
    main()