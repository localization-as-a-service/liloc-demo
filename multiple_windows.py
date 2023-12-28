# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import threading
import time
import os
import utils.fread as fread

CLOUD_NAME = "points"


def main():
    MultiWinApp().run()


class MultiWinApp:

    def __init__(self):
        self.is_done = False
        self.n_snapshots = 0
        self.cloud = None
        self.main_vis = None
        self.snapshot_pos = None

    def run(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.main_vis = o3d.visualization.O3DVisualizer("Open3D - Multi-Window Demo")
        self.main_vis.add_action("Take snapshot in new window", self.on_snapshot)
        self.main_vis.set_on_close(self.on_main_window_closing)

        app.add_window(self.main_vis)
        self.snapshot_pos = (self.main_vis.os_frame.x, self.main_vis.os_frame.y)

        threading.Thread(target=self.update_thread).start()

        app.run()

    def on_snapshot(self, vis):
        self.n_snapshots += 1
        self.snapshot_pos = (self.snapshot_pos[0] + 50,
                             self.snapshot_pos[1] + 50)
        title = "Open3D - Multi-Window Demo (Snapshot #" + str(
            self.n_snapshots) + ")"
        new_vis = o3d.visualization.O3DVisualizer(title)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        new_vis.add_geometry(CLOUD_NAME + " #" + str(self.n_snapshots),
                             self.cloud, mat)
        new_vis.reset_camera_to_default()
        bounds = self.cloud.get_axis_aligned_bounding_box()
        extent = bounds.get_extent()
        new_vis.setup_camera(60, bounds.get_center(),
                             bounds.get_center() + [0, 0, -3], [0, -1, 0])
        o3d.visualization.gui.Application.instance.add_window(new_vis)
        new_vis.os_frame = o3d.visualization.gui.Rect(self.snapshot_pos[0],
                                                      self.snapshot_pos[1],
                                                      new_vis.os_frame.width,
                                                      new_vis.os_frame.height)

    def on_main_window_closing(self):
        self.is_done = True
        return True  # False would cancel the close

    def update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.
        sequence_dir = "data/registration_sample/01"
        sequence_ts = fread.get_timstamps(sequence_dir, ".global.pcd")

        self.cloud = o3d.geometry.PointCloud()

        # pcd_data = o3d.data.DemoICPPointClouds()
        # self.cloud = o3d.io.read_point_cloud(pcd_data.paths[0])
        # bounds = self.cloud.get_axis_aligned_bounding_box()
        # extent = bounds.get_extent()

        def add_first_cloud():
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            self.main_vis.add_geometry(CLOUD_NAME, self.cloud, mat)
            self.main_vis.reset_camera_to_default()
            # self.main_vis.setup_camera(60, bounds.get_center(),
            #                            bounds.get_center() + [0, 0, -3],
            #                            [0, -1, 0])

        o3d.visualization.gui.Application.instance.post_to_main_thread(self.main_vis, add_first_cloud)

        for t in sequence_ts:
            pcd = o3d.io.read_point_cloud(os.path.join(sequence_dir, f"{t}.global.pcd"))
            pcd = pcd.voxel_down_sample(0.05)
            pcd.paint_uniform_color([1, 0.706, 0])

            self.cloud.points = pcd.points
            self.cloud.colors = pcd.colors

            def update_cloud():
                # Note: if the number of points is less than or equal to the
                #       number of points in the original object that was added,
                #       using self.scene.update_geometry() will be faster.
                #       Requires that the point cloud be a t.PointCloud.
                self.main_vis.remove_geometry(CLOUD_NAME)
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultUnlit"
                self.main_vis.add_geometry(CLOUD_NAME, self.cloud, mat)

        
            o3d.visualization.gui.Application.instance.post_to_main_thread(self.main_vis, update_cloud)


if __name__ == "__main__":
    main()
