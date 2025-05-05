import open3d as o3d
import numpy as np
from replay_realsense_tensor import read_RGB_D_folder
import time 
import cv2

imageStream = read_RGB_D_folder('realsense2',starting_index=150,depth_num=3,debug_mode=False)
while imageStream.has_next():
    start_cv2 = cv2.getTickCount()
    if imageStream.has_next():
        count, depth_image, color_image, t2cam_pcd, t2cam_pcd_cuda,vertex_map_np, vertex_map_gpu, normal_map_np , normal_map_gpu,mesh= imageStream.get_next_frame()
    #t2cam_pcd.to_legacy()
    end_cv2 = cv2.getTickCount()
    time_sec = (end_cv2-start_cv2)/cv2.getTickFrequency()
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    print("FPS:", time_sec)

# intrinsic = o3d.core.Tensor([[433.4320983886719,0,416.3398132324219],[0,432.88079833984375,238.8269500732422],[0,0,1]]).cuda()
# #print(np.asarray(intrinsic))
# intrinsic_nt = o3d.io.read_pinhole_camera_intrinsic("real_time_camera_intrinsic.json")
# #print(intrinsic_nt)

# if imageStream.has_next():
#     count, depth_image, color_image, rgbd_image, t2cam_pcd = imageStream.get_next_frame()
# if imageStream.has_next():
#     count1, depth_image1, color_image1, rgbd_image1, t2cam_pcd1 = imageStream.get_next_frame()

# t0 = time.time()
# depth_o3d = o3d.t.geometry.Image(depth_image).cuda()
# print(depth_image)
# color_o3d = o3d.t.geometry.Image(color_image).cuda()
# RGBD = o3d.t.geometry.RGBDImage(color_o3d,depth_o3d,10000)
# pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(RGBD,intrinsic)
# pcd_cpu = pcd.cpu()
# #pcd.estimate_normals() #bad for gpu
# #pcd.orient_normals_consistent_tangent_plane(k=10)
# #pcd.uniform_down_sample(every_k_points=10)
# pcd2 = pcd.clone()
# pcd2.translate(o3d.core.Tensor([0.01,0.1,0.01]))
# pcd2.estimate_normals()
# t1 = time.time()
# #filtered_depth = depth_o3d.filter_bilateral(kernel_size = 7, value_sigma= 10, dist_sigma = 20.0)
# vertex_map = depth_o3d.create_vertex_map(intrinsic)
# normal_map = vertex_map.create_normal_map()
# print((normal_map.as_tensor().cpu().numpy())[240,422]) #480, 848
# #o3d.visualization.draw([pcd,pcd2])
# # reg_res = o3d.t.pipelines.registration.icp(source = pcd,
# #                                            target = pcd2,
# #                                            max_correspondence_distance = 0.02,
# #                                            estimation_method = o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
# #                                            criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-08, relative_rmse=1.000000e-08, max_iteration=1000))

# # metric_params = o3d.t.geometry.MetricParameters()
# # metrics = pcd.compute_metrics(
# #     pcd2, [o3d.t.geometry.Metric.ChamferDistance],
# #     metric_params)

# # print(metrics.cpu().numpy())
# # np.testing.assert_allclose(
# #     metrics.cpu().numpy(),
# #     (0.22436734, np.sqrt(3) / 10, 100. / 8, 400. / 8, 700. / 8, 100.),
# #     rtol=1e-6)
# t2 = time.time()

# print(depth_o3d.is_cuda, color_o3d.is_cuda,RGBD.is_cuda)
# print("Time 1", t1-t0)
# print("Time 2", t2-t1)

# p = o3d.t.geometry.PointCloud()
# p.point.positions = vertex_map.as_tensor().cpu().reshape((-1, 3))
# p.point.normals = normal_map.as_tensor().cpu().reshape((-1, 3))
# # o3d.visualization.draw([p])

# legacy_pcd = p.to_legacy()
# o3d.visualization.draw_geometries([legacy_pcd], point_show_normal=True)

# t0 = time.time()
# depth_o3d = o3d.t.geometry.Image(depth_image)
# color_o3d = o3d.t.geometry.Image(color_image)
# RGBD = o3d.t.geometry.RGBDImage(color_o3d,depth_o3d,10000)
# pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(RGBD,intrinsic)
# pcd_cuda = pcd.cuda()
# pcd.uniform_down_sample(every_k_points=10)
# #pcd.estimate_normals()
# pcd2 = pcd.translate([0.01,0.1,0.01])
# pcd2.estimate_normals()
# t1 = time.time()


# # reg_res = o3d.t.pipelines.registration.icp(source = pcd,
# #                                            target = pcd2,
# #                                            max_correspondence_distance = 0.02,
# #                                            estimation_method = o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
# #                                            criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-08, relative_rmse=1.000000e-08, max_iteration=1000))

# # pcd2 = pcd.translate([0.1,0.1,0.1])
# # metric_params = o3d.t.geometry.MetricParameters(fscore_radius=o3d.utility.DoubleVector((0.01, 0.11, 0.15, 0.18)))
# # metrics = pcd.compute_metrics(
# #     pcd2, [o3d.t.geometry.Metric.FScore],
# #     metric_params)

# # print(metrics.cpu().numpy())
# # np.testing.assert_allclose(
# #     metrics.cpu().numpy(),
# #     (0.22436734, np.sqrt(3) / 10, 100. / 8, 400. / 8, 700. / 8, 100.),
# #     rtol=1e-6)
# t2 = time.time()

# print(depth_o3d.is_cuda, color_o3d.is_cuda,RGBD.is_cuda)
# print("Time 1", t1-t0)
# print("Time 2", t2-t1)

# t0 = time.time()
# depth_o3d = o3d.geometry.Image(depth_image)
# color_o3d = o3d.geometry.Image(color_image)
# RGBD = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d,depth_o3d,10000)
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(RGBD,intrinsic_nt)
# #pcd.estimate_normals()
# pcd.uniform_down_sample(every_k_points=10)
# pcd2 = pcd.translate([0.1,0.1,0.1])
# pcd2.estimate_normals()
# t1 = time.time()

# # reg_res = o3d.pipelines.registration.registration_icp(source = pcd,
# #                                            target = pcd2,
# #                                            max_correspondence_distance = 0.02,
# #                                            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(),
# #                                            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_rmse=1.000000e-08, max_iteration=1000))
# # print(reg_res)
# t2 = time.time()

# #print(depth_o3d.is_cuda, color_o3d.is_cuda,RGBD.is_cuda)
# print("Time 1", t1-t0)
# print("Time 2", t2-t1)

