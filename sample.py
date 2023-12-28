import requests
import time
import numpy as np
import cv2
import open3d


def main():
    try:
        response = requests.get("http://localhost:5000/health-check")
        print(response.text)

        response = requests.get("http://localhost:5000/start")
        print(response.text)

        time.sleep(5)

        # response = requests.get("http://localhost:5000/latest-depth-image")
        
        # data = response.json()

        # for k in data.keys():
        #     image = np.array(data[k]["image"], dtype=np.uint16)
        #     cv2.imshow(k, image)
        #     cv2.waitKey(0)

        response = requests.get(f"http://localhost:5000/global-pcd?timestamp={int(time.time() * 1000)}")

        data = response.json()
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(data))
        open3d.visualization.draw_geometries([pcd])

        # open3d.io.write_point_cloud("global.ply", pcd)
        

    finally:
        response = requests.get("http://localhost:5000/stop")

        print(response.text)


if __name__ == "__main__":
    main()