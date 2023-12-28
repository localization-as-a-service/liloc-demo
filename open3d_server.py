import open3d as o3d

if __name__ == "__main__":
    o3d.visualization.webrtc_server.enable_webrtc()
    
    pcd = o3d.io.read_point_cloud("data/registration_sample/01/1675753304363.global.pcd")
    pcd = pcd.voxel_down_sample(0.05)


    o3d.visualization.draw(pcd)