import numpy as np
import cv2
import open3d as o3d
import os

class read_RGB_D_folder:
    def __init__(self, folder, starting_index=0,depth_num=3,debug_mode=True):
        self.folder = folder
        self.depth_folder = os.path.join(folder, "depth_3_18")
        self.color_folder = os.path.join(folder, "color_3_18")
        self.depth_num = depth_num
        self.index = starting_index
        self.debug_mode = debug_mode
        self.depth_files = [os.path.join(self.depth_folder, f) for f in os.listdir(self.depth_folder) if f.endswith(".png")]
        self.color_files = [os.path.join(self.color_folder, f) for f in os.listdir(self.color_folder) if f.endswith(".jpg")]

    def has_next(self):
        return self.index < len(self.color_files)
    
    def get_next_frame(self):
        if self.debug_mode: print(self.index, self.depth_files[self.index],self.color_files[self.index])

        current_depth = np.array(o3d.io.read_image(self.depth_files[self.index]))
        current_color = np.array(o3d.io.read_image(self.color_files[self.index]))
        self.index += 1
        depth_scale = 0.1
        depth_image_scaled = current_depth * depth_scale
        depth_image_scaled = o3d.geometry.Image(depth_image_scaled.astype(np.float32))
        depth_o3d = o3d.geometry.Image(depth_image_scaled)
        color_o3d = o3d.geometry.Image(current_color)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, depth_trunc = 5.0)

        if self.depth_num == 3:
            intrinsic = o3d.io.read_pinhole_camera_intrinsic("real_time_camera_intrinsic.json")
        elif self.depth_num == 0:
            intrinsic = o3d.io.read_pinhole_camera_intrinsic("camera_intrinsic.json")

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        return current_depth, current_color, rgbd_image, pcd
