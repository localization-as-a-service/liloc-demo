import json
import open3d
import numpy as np

from PIL import Image
from dataclasses import dataclass


@dataclass
class DepthCameraParams:
    fx: float
    fy: float
    px: float
    py: float
    width: int
    height: int
    depth_scale: float
    intrinsics: open3d.camera.PinholeCameraIntrinsic
    
    def __init__(self, metadata_fname: str):
        with open(metadata_fname, "r") as f:
            metadata = json.load(f)
            self.fx = metadata["fx"]
            self.fy = metadata["fy"]
            self.px = metadata["px"]
            self.py = metadata["py"]
            self.width = metadata["width"]
            self.height = metadata["height"]
            self.depth_scale = metadata["depth_scale"]
            self.intrinsics = open3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.fx, self.fy, self.px, self.py)
            

class DepthCamera:
    
    @staticmethod
    def get_meshgrid(camera_params: DepthCameraParams):
        width, height = camera_params.intrinsics.width, camera_params.intrinsics.height
        fx, fy = camera_params.intrinsics.get_focal_length()
        cx, cy = camera_params.intrinsics.get_principal_point()

        x = (np.arange(width) - cx) / fx
        y = (np.arange(height) - cy) / fy

        return np.meshgrid(x, y)
    
    @staticmethod
    def read_depth_image(depth_image_fname: str):
        depth_image = Image.open(depth_image_fname).convert("I")
        return np.array(depth_image)
            
    def __init__(self, camera_params: DepthCameraParams) -> None:
        self.camera_params = camera_params
        self.meshgrid = DepthCamera.get_meshgrid(camera_params)

    def depth_image_to_point_cloud(self, depth_image: np.ndarray):
        z = depth_image / self.camera_params.depth_scale
        
        x, y = self.meshgrid

        xyz = np.dstack((x * z, y * z, z))
        xyz = xyz[z > 0]
        
        xpcd = open3d.geometry.PointCloud()
        xpcd.points = open3d.utility.Vector3dVector(xyz)

        return xpcd