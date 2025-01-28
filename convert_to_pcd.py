import cv2
import robotpy_apriltag
import numpy as np
import open3d as o3d
import copy
import time

t0= time.time()
print(t0)

intrinsic = o3d.io.read_pinhole_camera_intrinsic( "camera_intrinsic.json")

source_color = o3d.io.read_image(f"000187.jpg")
source_depth = o3d.io.read_image(f"000187.png")

source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            source_color, source_depth, depth_trunc = 6)
source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, intrinsic)
o3d.io.write_point_cloud("./T2CAM_2.pcd",source_pcd)
#source_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
source_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

