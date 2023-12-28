import numpy as np
import pandas as pd
import os
import tqdm
import open3d
import time
from threading import Thread
from multiprocessing import Process

import utils.pointcloud as pointcloud
import utils.fread as fread
import utils.pointcloud as pointcloud
import utils.FCGF as FCGF

from utils.config import Config
from time import sleep



class Visualizer(Thread):
    def __init__(self):
        super(Visualizer, self).__init__()
        self.vis = open3d.visualization.Visualizer()
        self.pcd = None
        self.running = True
    
    def update_geometry(self, geometry):
        self.pcd.points = geometry.points
        self.pcd.colors = geometry.colors

        # self.vis.update_geometry()
        # self.vis.poll_events()
        # self.vis.update_renderer()

    def add_geometry(self, geometry):
        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = geometry.points
        self.vis.add_geometry(self.pcd)

    def set_geometry(self, geometry):
        if self.pcd is not None:
            self.update_geometry(geometry)
        else:
            self.add_geometry(geometry)

    def run(self):
        self.vis.create_window()

        while self.running:
            self.vis.update_geometry()
            self.vis.poll_events()
            self.vis.update_renderer()

            time.sleep(0.001)

        self.vis.destroy_window()

    def stop(self):
        self.running = False


def main():
    sequence_dir = "data/registration_sample/01"
    sequence_ts = fread.get_timstamps(sequence_dir, ".global.pcd")

    # vis = open3d.visualization.Visualizer()
    # vis.create_window()

    vis = Visualizer()
    vis.start()

    # pcd = open3d.geometry.PointCloud()
    # vis.add_geometry(pcd)
    # pcd = None

    for t in sequence_ts:
        print(t)
        x = open3d.io.read_point_cloud(os.path.join(sequence_dir, f"{t}.global.pcd"))

        # if not pcd:
        #     pcd = x
        #     vis.add_geometry(pcd)
        # else:
        #     pcd.points = x.points
        
        # vis.update_geometry()
        # vis.poll_events()
        # vis.update_renderer()
        
        # sleep(0.01)
        vis.set_geometry(x)

    vis.stop()


if __name__ == "__main__":
    main()
    
    
        
    